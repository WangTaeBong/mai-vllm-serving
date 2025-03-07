#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mai-vllm-serving의 분산 처리 구현
다중 GPU 환경에서 vLLM을 효율적으로 분산 처리하기 위한 기능
"""

import asyncio
import datetime
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union, Any, AsyncGenerator

import torch
import torch.distributed as dist
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from mai_vllm_serving.engine import RequestConfig
from mai_vllm_serving.monitoring.metrics import get_metrics_collector
from mai_vllm_serving.monitoring.profiler import profile, profile_block
from mai_vllm_serving.utils.config import get_config
from mai_vllm_serving.utils.logging_utils import (
    setup_logging,
    TimingContext
)

# 설정 객체 가져오기
config = get_config()

# 로깅 초기화
logger = setup_logging(
    service_name="mai-vllm-serving-distributed",
    log_level=config.logging.level,
    use_json=config.logging.json,
    log_file=config.logging.file
)


@dataclass
class DistributedConfig:
    """분산 처리 설정을 위한 데이터 클래스"""
    world_size: int = config.distributed.world_size
    rank: int = 0  # 현재 프로세스의 순위 (0부터 시작)
    local_rank: int = 0  # 현재 노드 내에서의 순위
    master_addr: str = config.distributed.master_addr
    master_port: str = config.distributed.master_port
    backend: str = config.distributed.backend
    init_method: Optional[str] = None  # 초기화 방법 (TCP, 파일 등)
    timeout: int = config.distributed.timeout

    @classmethod
    def from_env(cls) -> 'DistributedConfig':
        """
        환경 변수에서 분산 처리 설정 가져오기

        Returns:
            분산 처리 설정
        """
        world_size = int(os.environ.get("WORLD_SIZE", str(config.distributed.world_size)))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        master_addr = os.environ.get("MASTER_ADDR", config.distributed.master_addr)
        master_port = os.environ.get("MASTER_PORT", config.distributed.master_port)
        backend = os.environ.get("DIST_BACKEND", config.distributed.backend)
        if not torch.cuda.is_available() and backend == "nccl":
            logger.warning("CUDA not available. Changing backend from nccl to gloo")
            backend = "gloo"
        init_method = os.environ.get("INIT_METHOD", None)
        timeout = int(os.environ.get("DIST_TIMEOUT", str(config.distributed.timeout)))

        return cls(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
            backend=backend,
            init_method=init_method,
            timeout=timeout
        )

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)


class DistributedVLLMEngine:
    """
    분산 vLLM 엔진

    다중 GPU 환경에서 vLLM을 효율적으로 분산 처리하기 위한 클래스
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 dist_config: Optional[DistributedConfig] = None,
                 tensor_parallel_size: Optional[int] = None,
                 pipeline_parallel_size: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 max_model_len: Optional[int] = None,
                 max_num_seqs: Optional[int] = None,
                 max_num_batched_tokens: Optional[int] = None,
                 quantization: Optional[str] = None,
                 swap_space: Optional[int] = None,
                 trust_remote_code: Optional[bool] = None,
                 dtype: Optional[str] = None,
                 enforce_eager: Optional[bool] = None):
        """
        분산 vLLM 엔진 초기화

        Args:
            model_name: 허깅페이스 모델 ID 또는 로컬 경로
            dist_config: 분산 처리 설정
            tensor_parallel_size: 텐서 병렬 처리에 사용할 GPU 수
            pipeline_parallel_size: 파이프라인 병렬 처리에 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률 (0.0 ~ 1.0)
            max_model_len: 최대 모델 ~
            max_num_seqs: 최대 배치 시퀀스 수
            max_num_batched_tokens: 배치당 최대 토큰 수
            quantization: 양자화 방식 (None, "awq", "gptq", "squeezellm")
            swap_space: CPU 스왑 공간 크기 (GB)
            trust_remote_code: 원격 코드 신뢰 여부
            dtype: 연산 데이터 타입 ("auto", "float16", "bfloat16", "float32")
            enforce_eager: PyTorch eager 모드 강제 실행 여부
        """
        # 분산 설정 초기화
        self.dist_config = dist_config or DistributedConfig.from_env()
        self.rank = self.dist_config.rank
        self.world_size = self.dist_config.world_size
        self.local_rank = self.dist_config.local_rank
        self.is_master = self.rank == 0

        # 모델 이름 설정
        self.model_name = model_name or config.model.name
        if not self.model_name:
            raise ValueError("모델 이름이 지정되지 않았습니다. model_name 매개변수 또는 설정에서 지정해야 합니다.")

        # 인자 검증 및 기본값 설정
        self._validate_and_set_defaults(
            tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization,
            max_model_len, max_num_seqs, max_num_batched_tokens, quantization,
            swap_space, trust_remote_code, dtype, enforce_eager
        )

        # GPU 설정
        self.num_gpus = self._setup_gpu()

        # 분산 엔진 설정 초기화
        self.tensor_parallel_size = self._compute_tensor_parallel_size(tensor_parallel_size)
        self.pipeline_parallel_size = pipeline_parallel_size or config.engine.pipeline_parallel_size

        # 스레드 안전성을 위한 락
        self._request_lock = asyncio.Lock()

        # 엔진 인자 설정
        engine_args = self._create_engine_args()

        # 엔진 초기화
        if self.is_master:
            self._log_initialization_params()

        # 엔진 초기화 시간 측정
        with TimingContext(logger if self.is_master else None, "Engine initialization") as timing:
            if not dist.is_initialized():
                self._init_distributed()

            with profile_block("distributed_engine_init"):  # 프로파일링 블록 추가
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)

                if self.is_master:
                    logger.info(f"Engine initialization completed in {timing.duration:.2f} seconds")

        # 요청 처리 관련 변수 초기화
        self.start_time = time.time()
        self.active_requests = 0
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.request_metrics = {}  # 요청별 메트릭 저장

        # 분산 동기화
        self._sync_distributed()

    def _validate_and_set_defaults(self,
                                   tensor_parallel_size, pipeline_parallel_size,
                                   gpu_memory_utilization, max_model_len, max_num_seqs,
                                   max_num_batched_tokens, quantization,
                                   swap_space, trust_remote_code,
                                   dtype, enforce_eager):
        """설정값 검증 및 기본값 설정"""
        # 텐서 병렬 크기 검증은 별도 메서드에서 수행
        self._validate_tensor_parallel_size(tensor_parallel_size)

        # 파이프라인 병렬 크기
        self.pipeline_parallel_size = pipeline_parallel_size or config.engine.pipeline_parallel_size
        if self.pipeline_parallel_size < 1:
            logger.warning(f"잘못된 pipeline_parallel_size: {self.pipeline_parallel_size}. 1로 설정합니다.")
            self.pipeline_parallel_size = 1

        # GPU 메모리 사용률
        self.gpu_memory_utilization = gpu_memory_utilization or config.engine.gpu_memory_utilization
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            logger.warning(f"잘못된 gpu_memory_utilization: {self.gpu_memory_utilization}. 0.9로 설정합니다.")
            self.gpu_memory_utilization = 0.9

        # 최대 모델 ~
        self.max_model_len = max_model_len or config.engine.max_model_len
        if self.max_model_len < 1:
            logger.warning(f"잘못된 max_model_len: {self.max_model_len}.")

        # 최대 시퀀스 수
        self.max_num_seqs = max_num_seqs or config.engine.max_num_seqs
        if self.max_num_seqs < 1:
            logger.warning(f"잘못된 max_num_seqs: {self.max_num_seqs}. 256으로 설정합니다.")
            self.max_num_seqs = 256

        # 최대 배치 토큰 수
        self.max_num_batched_tokens = max_num_batched_tokens or config.engine.max_num_batched_tokens
        if self.max_num_batched_tokens < 1:
            logger.warning(f"잘못된 max_num_batched_tokens: {self.max_num_batched_tokens}. 8192로 설정합니다.")
            self.max_num_batched_tokens = 8192

        # 스왑 공간
        self.swap_space = swap_space or config.engine.swap_space
        if self.swap_space < 0:
            logger.warning(f"잘못된 swap_space: {self.swap_space}. 4로 설정합니다.")
            self.swap_space = 4

        # 원격 코드 신뢰 여부
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else config.model.trust_remote_code

        # 데이터 타입
        valid_dtypes = {"auto", "float16", "bfloat16", "float32"}
        self.dtype = dtype or config.engine.dtype
        if self.dtype not in valid_dtypes:
            logger.warning(f"잘못된 dtype: {self.dtype}. 'auto'로 설정합니다.")
            self.dtype = "auto"

        # Eager 모드 강제 실행 여부
        self.enforce_eager = enforce_eager if enforce_eager is not None else config.engine.enforce_eager

        # 양자화 설정
        valid_quantization = {None, "awq", "gptq", "squeezellm"}
        if quantization is None and config.quantization.enabled:
            self.quantization = config.quantization.method
        else:
            self.quantization = quantization

        if self.quantization not in valid_quantization:
            logger.warning(f"지원하지 않는 quantization: {self.quantization}. None으로 설정합니다.")
            self.quantization = None

    def _setup_gpu(self):
        """GPU 설정 및 확인"""
        if torch.cuda.is_available():
            # 현재 프로세스가 사용할 GPU 지정
            torch.cuda.set_device(self.local_rank)

            # 사용 가능한 GPU 수 확인
            num_gpus = torch.cuda.device_count()

            if self.is_master:
                logger.info(f"Using distributed mode with {self.world_size} processes")
                logger.info(
                    f"Process {self.rank} using GPU {self.local_rank}: {torch.cuda.get_device_name(self.local_rank)}")
                logger.info(f"Number of GPUs per node: {num_gpus}")
            return num_gpus
        else:
            logger.warning("No GPU available, using CPU. Performance will be significantly degraded.")
            return 0

    def _compute_tensor_parallel_size(self, tensor_parallel_size):
        """텐서 병렬 크기 계산 및 검증"""
        # 텐서 병렬 크기가 설정되지 않은 경우, 적절한 값 계산
        if tensor_parallel_size is None:
            # 텐서 병렬 크기 = min(노드당 GPU 수, 8)
            tensor_parallel_size = min(self.num_gpus if self.num_gpus > 0 else 1, 8)
            logger.info(f"자동 설정된 텐서 병렬 크기: {tensor_parallel_size}")

        # 병렬 크기 유효성 검사 수행
        tensor_parallel_size = self._validate_tensor_parallel_size(tensor_parallel_size)
        return tensor_parallel_size

    def _validate_tensor_parallel_size(self, tensor_parallel_size):
        """텐서 병렬 크기 유효성 검사"""
        # 텐서 병렬 크기가 정수인지 확인
        try:
            tensor_parallel_size = int(tensor_parallel_size)
        except (ValueError, TypeError):
            logger.warning(f"텐서 병렬 크기는 정수여야 합니다. 입력값: {tensor_parallel_size}, 기본값 1로 설정합니다.")
            return 1

        # 텐서 병렬 크기가 1보다 작은 경우
        if tensor_parallel_size < 1:
            logger.warning(f"텐서 병렬 크기는 최소 1 이상이어야 합니다. 입력값: {tensor_parallel_size}, 기본값 1로 설정합니다.")
            return 1

        # 텐서 병렬 크기가 GPU 수보다 큰 경우
        if 0 < self.num_gpus < tensor_parallel_size:
            logger.warning(
                f"요청한 텐서 병렬 크기({tensor_parallel_size})가 사용 가능한 GPU 수({self.num_gpus})보다 큽니다.")
            logger.warning(f"텐서 병렬 크기를 {self.num_gpus}로 조정합니다.")
            return self.num_gpus

        # 텐서 병렬 크기가 2의 거듭제곱인지 확인
        if tensor_parallel_size > 1 and (tensor_parallel_size & (tensor_parallel_size - 1)) != 0:
            # 가장 가까운 2의 거듭제곱 값 찾기
            next_power = 2 ** (tensor_parallel_size.bit_length() - 1)
            if next_power * 2 <= self.num_gpus:
                next_power *= 2

            logger.warning(
                f"최적의 성능을 위해 텐서 병렬 크기는 2의 거듭제곱(2, 4, 8...)이어야 합니다. "
                f"입력값: {tensor_parallel_size}, {next_power}로 조정합니다.")
            return next_power

        # 텐서 병렬 크기가 너무 큰 경우 (일반적으로 16 이상은 성능 저하 가능성)
        if tensor_parallel_size > 16:
            logger.warning(
                f"텐서 병렬 크기가 매우 큽니다({tensor_parallel_size}). "
                f"이는 통신 오버헤드로 인해 성능 저하를 초래할 수 있습니다.")

        # 텐서 병렬 크기가 세계 크기(world_size)에 나누어 떨어지는지 확인
        if self.world_size % tensor_parallel_size != 0:
            logger.warning(
                f"텐서 병렬 크기({tensor_parallel_size})가 세계 크기({self.world_size})의 약수가 아닙니다. "
                f"이는 일부 프로세스가 사용되지 않을 수 있습니다.")

        # 모든 검증을 통과한 경우
        logger.info(f"검증된 텐서 병렬 크기: {tensor_parallel_size}")
        return tensor_parallel_size

    def _create_engine_args(self):
        """vLLM 엔진 인자 생성"""
        return AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            quantization=self.quantization,
            swap_space=self.swap_space,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            block_size=config.engine.block_size
        )

    def _log_initialization_params(self):
        """초기화 매개변수 로깅"""
        logger.info(f"Initializing distributed vLLM engine with {self.model_name}")
        logger.info(f"Tensor parallel size: {self.tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {self.pipeline_parallel_size}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")
        logger.info(f"Max model length: {self.max_model_len}")
        logger.info(f"Max num sequences: {self.max_num_seqs}")
        logger.info(f"Max num batched tokens: {self.max_num_batched_tokens}")
        logger.info(f"Quantization: {self.quantization}")
        logger.info(f"Swap space: {self.swap_space} GB")
        logger.info(f"Dtype: {self.dtype}")
        logger.info(f"Enforce eager: {self.enforce_eager}")

    def _init_distributed(self):
        """분산 환경 초기화"""
        # 이미 초기화된 경우 스킵
        if dist.is_initialized():
            logger.info("Distributed environment is already initialized")
            return

        # 분산 환경 변수 설정
        os.environ["MASTER_ADDR"] = self.dist_config.master_addr
        os.environ["MASTER_PORT"] = self.dist_config.master_port
        os.environ["WORLD_SIZE"] = str(self.dist_config.world_size)
        os.environ["RANK"] = str(self.dist_config.rank)
        os.environ["LOCAL_RANK"] = str(self.dist_config.local_rank)

        # 분산 환경 초기화 시도
        with TimingContext(logger, "Distributed init") as timing:
            try:
                with profile_block("distributed_init"):  # 프로파일링 블록 추가
                    logger.info(
                        f"Initializing distributed environment (rank={self.rank}, world_size={self.world_size})")
                    dist.init_process_group(
                        backend=self.dist_config.backend,
                        init_method=self._get_init_method(),
                        world_size=self.world_size,
                        rank=self.rank,
                        timeout=datetime.timedelta(seconds=self.dist_config.timeout)
                    )
                logger.info(f"Distributed environment initialized successfully in {timing.duration:.2f}s")
            except Exception as e:
                logger.error(f"Failed to initialize distributed environment: {str(e)}", exc_info=True)
                raise

    def _get_init_method(self) -> str:
        """분산 초기화 방법 문자열 생성"""
        if self.dist_config.init_method:
            return self.dist_config.init_method

        # 기본값: TCP 초기화 방법
        return f"tcp://{self.dist_config.master_addr}:{self.dist_config.master_port}"

    def _sync_distributed(self):
        """분산 환경 동기화"""
        if dist.is_initialized():
            if self.is_master:
                logger.info("Synchronizing distributed processes...")

            with TimingContext(logger if self.is_master else None, "Distributed synchronization") as timing:
                try:
                    dist.barrier()
                    if self.is_master:
                        logger.info(f"Distributed processes synchronized in {timing.duration:.2f}s")
                except Exception as e:
                    logger.error(f"Error during distributed synchronization: {str(e)}", exc_info=True)
                    raise

    @profile
    async def generate(self, req_config: RequestConfig) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        텍스트 생성 요청 처리

        Args:
            req_config: 생성 요청 설정

        Returns:
            생성된 텍스트와 관련 메타데이터 또는 스트리밍 응답
        """
        # 요청 ID 생성 또는 사용
        if req_config.request_id is None:
            req_config.request_id = str(uuid.uuid4())

        # 입력 검증
        self._validate_request_config(req_config)

        # 요청 메트릭 초기화
        request_id = req_config.request_id
        if request_id not in self.request_metrics:
            self.request_metrics[request_id] = {
                "start_time": time.time(),
                "status": "pending",
                "tokens_generated": 0,
                "prompt_tokens": 0
            }

        # 메트릭 수집기 가져오기 (마스터 노드에서만 메트릭 수집)
        metrics_collector = get_metrics_collector() if self.is_master else None

        # 마스터 노드의 경우 메트릭 수집 시작
        if self.is_master and metrics_collector:
            metrics_collector.start_request(request_id)

        # 추론 시작 로깅
        logger.info(f"Request {request_id} received: prompt length={len(req_config.prompt)}, "
                    f"max_tokens={req_config.max_tokens}, temperature={req_config.temperature}")

        # 스레드 안전을 위한 락 획득 (요청 카운터 업데이트 시)
        async with self._request_lock:
            # 활성 요청 수 증가
            self.active_requests += 1
            self.total_requests += 1
            self.request_metrics[request_id]["status"] = "processing"

        # 타이밍 컨텍스트 시작
        timing = TimingContext(logger, f"Request {request_id} processing")

        # 요청 처리 및 오류 처리
        try:
            # SamplingParams 생성
            engine_request = self._create_engine_request(req_config)
            sampling_params = engine_request.to_sampling_params()

            # 마스터 노드의 경우 요청 처리 시작 메트릭 기록
            if self.is_master and metrics_collector:
                metrics_collector.start_processing(request_id)

            # 스트리밍 모드인 경우
            if req_config.stream:
                logger.debug(f"Starting streaming response for request {request_id}")
                with profile_block(f"stream_request_{request_id}"):
                    return self._stream_generator(engine_request, sampling_params, timing, metrics_collector)

            # 일반 모드인 경우 - 타이밍 컨텍스트 시작
            with timing:
                result = None
                with profile_block(f"generate_request_{engine_request.request_id}"):
                    async for res in self.engine.generate(
                            prompt=engine_request.prompt,
                            sampling_params=sampling_params,
                            request_id=engine_request.request_id):
                        result = res

            # 결과 확인
            if result is None:
                error_msg = f"No result generated for request {request_id}"
                logger.error(error_msg)
                if self.is_master and metrics_collector:
                    metrics_collector.fail_request(request_id, error_msg)
                raise RuntimeError(error_msg)

            # 결과 파싱
            output = result.outputs[0]

            # 추론 통계 계산
            prompt_tokens = len(result.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            total_tokens = prompt_tokens + completion_tokens
            tokens_per_second = completion_tokens / timing.duration if timing.duration > 0 else 0

            # 메트릭 업데이트
            async with self._request_lock:
                self.total_tokens_generated += completion_tokens
                self.request_metrics[request_id].update({
                    "status": "completed",
                    "tokens_generated": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "end_time": time.time(),
                    "duration": timing.duration
                })

            # 마스터 노드의 경우 메트릭 수집기에 완료 기록
            if self.is_master and metrics_collector:
                metrics_collector.complete_request(
                    request_id,
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens
                )

            # 통계 로깅
            if self.is_master:
                logger.info(
                    f"Request {request_id} completed: {prompt_tokens} prompt tokens, "
                    f"{completion_tokens} completion tokens in {timing.duration:.3f}s "
                    f"({tokens_per_second:.2f} tokens/sec)"
                )

            return {
                "id": request_id,
                "generated_text": output.text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "performance": {
                    "inference_time": f"{timing.duration:.3f}",
                    "tokens_per_second": f"{tokens_per_second:.2f}"
                },
                "finish_reason": output.finish_reason
            }

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
            # 마스터 노드의 경우 메트릭 수집기에 실패 기록
            if self.is_master and metrics_collector:
                metrics_collector.fail_request(request_id, str(e))

            # 메트릭 업데이트
            async with self._request_lock:
                self.request_metrics[request_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.time()
                })
            raise

        finally:
            # 활성 요청 수 감소 (lock 보호)
            async with self._request_lock:
                self.active_requests -= 1
                # 오래된 요청 메트릭 정리 (최대 1000개 유지)
                if len(self.request_metrics) > 1000:
                    # 완료 또는 실패한 가장 오래된 요청부터 삭제
                    completed_requests = [
                        req_id for req_id, metrics in self.request_metrics.items()
                        if metrics.get("status") in ("completed", "failed")
                    ]
                    # 삭제할 요청 수 계산 (전체의 20%)
                    remove_count = min(len(completed_requests), 200)
                    for req_id in sorted(completed_requests,
                                         key=lambda x: self.request_metrics[x].get("end_time", 0))[:remove_count]:
                        del self.request_metrics[req_id]

    @classmethod
    def _validate_request_config(cls, req_config: RequestConfig):
        """요청 설정 검증"""
        if not req_config.prompt:
            raise ValueError("프롬프트가 비어 있습니다.")

        if req_config.max_tokens is not None and req_config.max_tokens <= 0:
            logger.warning(f"잘못된 max_tokens: {req_config.max_tokens}. 기본값 사용.")
            req_config.max_tokens = config.inference.max_tokens

        if req_config.temperature is not None and req_config.temperature < 0:
            logger.warning(f"잘못된 temperature: {req_config.temperature}. 기본값 사용.")
            req_config.temperature = config.inference.temperature

    @classmethod
    def _create_engine_request(cls, req_config: RequestConfig) -> RequestConfig:
        """엔진 요청 객체 생성"""
        return RequestConfig(
            prompt=req_config.prompt,
            # 아래 값들은 요청에 있을 경우 그 값을 사용, 없으면 config의 기본값 사용
            max_tokens=req_config.max_tokens if req_config.max_tokens is not None else config.inference.max_tokens,
            temperature=req_config.temperature if req_config.temperature is not None else config.inference.temperature,
            top_p=req_config.top_p if req_config.top_p is not None else config.inference.top_p,
            top_k=req_config.top_k if req_config.top_k is not None else config.inference.top_k,
            frequency_penalty=req_config.frequency_penalty if req_config.frequency_penalty is not None else config.inference.frequency_penalty,
            presence_penalty=req_config.presence_penalty if req_config.presence_penalty is not None else config.inference.presence_penalty,
            repetition_penalty=req_config.repetition_penalty if req_config.repetition_penalty is not None else config.inference.repetition_penalty,
            no_repeat_ngram_size=req_config.no_repeat_ngram_size if req_config.no_repeat_ngram_size is not None else config.inference.no_repeat_ngram_size,
            stop=req_config.stop,
            stream=req_config.stream,
            request_id=req_config.request_id,
            seed=config.inference.seed
        )

    async def _stream_generator(self,
                                engine_request: RequestConfig,
                                sampling_params: SamplingParams,
                                timing: TimingContext,
                                metrics_collector=None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 생성 요청 처리를 위한 제너레이터

        Args:
            engine_request: 요청 객체
            sampling_params: SamplingParams 객체
            timing: 타이밍 컨텍스트
            metrics_collector: 메트릭 수집기 (없으면 None)

        Yields:
            생성된 텍스트 조각
        """
        request_id = engine_request.request_id
        # tokens_generated = 0
        # prompt_tokens = 0
        stream_finished = False

        # 타이밍 시작
        timing.__enter__()

        try:
            async for result in self.engine.generate(
                    prompt=engine_request.prompt,
                    sampling_params=sampling_params,
                    request_id=request_id):
                output = result.outputs[0]
                tokens_generated = len(output.token_ids)
                prompt_tokens = len(result.prompt_token_ids)

                # 토큰 제한 확인
                if engine_request.max_tokens and tokens_generated >= engine_request.max_tokens:
                    logger.debug(f"Request {request_id} reached max tokens limit: {engine_request.max_tokens}")

                # 응답이 완료된 경우 (finish_reason이 None이 아닌 경우)
                is_finished = output.finish_reason is not None

                # 스트림이 완료되면 (한 번만 실행)
                if is_finished and not stream_finished:
                    stream_finished = True

                    # 통계 업데이트 (lock 보호)
                    async with self._request_lock:
                        self.total_tokens_generated += tokens_generated
                        self.request_metrics[request_id].update({
                            "status": "completed",
                            "tokens_generated": tokens_generated,
                            "prompt_tokens": prompt_tokens,
                            "end_time": time.time(),
                            "duration": timing.duration
                        })

                    # 마스터 노드의 경우 메트릭 수집기에 완료 기록
                    if self.is_master and metrics_collector:
                        metrics_collector.complete_request(
                            request_id,
                            completion_tokens=tokens_generated,
                            prompt_tokens=prompt_tokens
                        )

                    if self.is_master:
                        tokens_per_second = tokens_generated / timing.duration if timing.duration > 0 else 0
                        logger.info(
                            f"Streaming request {request_id} completed: {tokens_generated} tokens in "
                            f"{timing.duration:.3f}s ({tokens_per_second:.2f} tokens/sec)"
                        )

                # 결과 반환
                yield {
                    "id": request_id,
                    "generated_text": output.text,
                    "finished": is_finished,
                    "finish_reason": output.finish_reason
                }

        except Exception as e:
            logger.error(f"Error in stream generation for request {request_id}: {str(e)}", exc_info=True)

            # 메트릭 업데이트 (lock 보호)
            async with self._request_lock:
                self.request_metrics[request_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.time()
                })

            # 마스터 노드의 경우 메트릭 수집기에 실패 기록
            if self.is_master and metrics_collector:
                metrics_collector.fail_request(request_id, str(e))

            # 오류 정보 반환
            yield {
                "id": request_id,
                "error": str(e),
                "finished": True
            }

        finally:
            # 타이밍 종료
            timing.__exit__(None, None, None)

            # 활성 요청 수 감소 (lock 보호)
            # 스트리밍이 완료될 때만 한 번 감소
            async with self._request_lock:
                # 이미 처리 중인 요청이면 카운터 감소
                if self.request_metrics[request_id]["status"] == "processing":
                    self.active_requests -= 1
