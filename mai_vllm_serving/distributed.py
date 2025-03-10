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
    get_logger,
    TimingContext,
    with_request_context
)

# 설정 객체 가져오기
config = get_config()

# 성능 모드 여부 확인
performance_mode = config.logging.log_performance_mode

# 구조화된 로거 초기화
logger = get_logger("distributed", production_mode=performance_mode)


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
            logger.warning(
                "CUDA not available. Changing backend from nccl to gloo",
                context={
                    "distributed": {
                        "backend_change": {"from": "nccl", "to": "gloo"},
                        "reason": "cuda_not_available"
                    }
                }
            )
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

        # 분산 초기화 로깅 - 구조화된 컨텍스트 사용
        dist_context = {
            "distributed": {
                "rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
                "master_addr": self.dist_config.master_addr,
                "master_port": self.dist_config.master_port,
                "backend": self.dist_config.backend,
                "is_master": self.is_master
            }
        }

        # 성능 모드이면서 마스터 노드가 아닌 경우 로깅 최소화
        if performance_mode and not self.is_master:
            # 마스터가 아닌 노드는 경량 로깅
            logger.debug("Initializing distributed worker node",
                         context={
                             "distributed": {
                                 "rank": self.rank,
                                 "is_master": False
                             }
                         })
        else:
            # 마스터 노드에서만 INFO 레벨 로깅
            if self.is_master:
                logger.info("Initializing distributed vLLM engine", context=dist_context)
            else:
                logger.debug("Initializing distributed vLLM engine worker", context=dist_context)

        # 모델 이름 설정
        self.model_name = model_name or config.model.name
        if not self.model_name:
            error_msg = "Model name not specified. Must be provided in parameters or configuration."
            logger.error(error_msg, context={"initialization": {"status": "failed"}})
            raise ValueError(error_msg)

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
        with TimingContext(logger if self.is_master else None, "Engine initialization",
                           log_threshold=0.5 if performance_mode else None) as timing:
            if not dist.is_initialized():
                self._init_distributed()

            with profile_block("distributed_engine_init"):  # 프로파일링 블록 추가
                try:
                    self.engine = AsyncLLMEngine.from_engine_args(engine_args)

                    if self.is_master:
                        logger.info(
                            "Engine initialization completed",
                            context={
                                "initialization": {
                                    "status": "success",
                                    "duration_seconds": timing.duration
                                }
                            }
                        )
                except Exception as e:
                    logger.error(
                        "Failed to initialize engine",
                        context={
                            "initialization": {
                                "status": "failed",
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        },
                        exc_info=True
                    )
                    raise

        # 요청 처리 관련 변수 초기화
        self.start_time = time.time()
        self.active_requests = 0
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.request_metrics = {}  # 요청별 메트릭 저장

        # 분산 동기화
        self._sync_distributed()

    @classmethod
    def shutdown(cls):
        """분산 엔진 종료 및 리소스 정리"""
        # 성능 모드에서는 로깅 최소화
        if not performance_mode:
            logger.info("Shutting down distributed environment", context={
                "shutdown": {
                    "component": "distributed_engine",
                    "status": "starting"
                }
            })
        else:
            logger.info("Shutting down distributed environment")

        if dist.is_initialized():
            try:
                # 타임아웃이 있는 barrier 사용
                timeout = datetime.timedelta(seconds=30)
                try:
                    logger.debug("Waiting for all processes at barrier", context=shutdown_context)
                    dist.barrier(timeout=timeout)
                    logger.info(
                        "All processes reached barrier",
                        context={"shutdown": {"barrier": "success"}}
                    )
                except Exception as e:
                    logger.warning(
                        "Error reaching barrier, continuing with shutdown",
                        context={
                            "shutdown": {
                                "barrier": "failed",
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        }
                    )

                # CUDA 캐시 정리
                if torch.cuda.is_available():
                    with TimingContext(logger, "CUDA cache cleanup") as timing:
                        torch.cuda.empty_cache()
                    logger.info(
                        "CUDA cache cleared",
                        context={"shutdown": {"cuda_cleanup": "success", "duration": timing.duration}}
                    )

                # 프로세스 그룹 정리
                logger.debug("Destroying process group", context=shutdown_context)
                dist.destroy_process_group()
                logger.info(
                    "Distributed process group successfully destroyed",
                    context={"shutdown": {"process_group": "success"}}
                )

            except Exception as e:
                logger.error(
                    "Error during distributed shutdown",
                    context={
                        "shutdown": {
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    },
                    exc_info=True
                )
                # 중요: 오류가 발생해도 기본 정리는 시도
                try:
                    dist.destroy_process_group()
                except Exception as inner_e:
                    logger.warning(
                        "Failed to destroy process group during recovery",
                        context={
                            "shutdown": {
                                "recovery_attempt": "failed",
                                "error_type": type(inner_e).__name__
                            }
                        }
                    )
            finally:
                # 환경 변수 정리
                env_vars_to_clean = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
                cleaned_vars = []

                for var in env_vars_to_clean:
                    if var in os.environ:
                        del os.environ[var]
                        cleaned_vars.append(var)

                if cleaned_vars:
                    logger.info(
                        "Cleaned environment variables",
                        context={"shutdown": {"cleaned_vars": cleaned_vars}}
                    )

                logger.info(
                    "Distributed shutdown completed",
                    context={"shutdown": {"status": "completed"}}
                )

    def _validate_and_set_defaults(self,
                                   tensor_parallel_size, pipeline_parallel_size,
                                   gpu_memory_utilization, max_model_len, max_num_seqs,
                                   max_num_batched_tokens, quantization,
                                   swap_space, trust_remote_code,
                                   dtype, enforce_eager):
        """설정값 검증 및 기본값 설정"""
        # 구조화된 컨텍스트 준비 - 성능 모드에서는 생략
        validation_context = {} if performance_mode else {"validation": {"component": "distributed_engine"}}

        # 텐서 병렬 크기 검증은 별도 메서드에서 수행
        self._validate_tensor_parallel_size(tensor_parallel_size)

        # 파이프라인 병렬 크기
        self.pipeline_parallel_size = pipeline_parallel_size or config.engine.pipeline_parallel_size
        if self.pipeline_parallel_size < 1:
            logger.warning(
                f"Invalid pipeline_parallel_size: {self.pipeline_parallel_size}. Setting to 1",
                context={
                    "validation": {
                        "parameter": "pipeline_parallel_size",
                        "invalid_value": self.pipeline_parallel_size,
                        "corrected_value": 1
                    }
                }
            )
            self.pipeline_parallel_size = 1

        # GPU 메모리 사용률
        self.gpu_memory_utilization = gpu_memory_utilization or config.engine.gpu_memory_utilization
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            logger.warning(
                f"Invalid gpu_memory_utilization: {self.gpu_memory_utilization}. Setting to 0.9",
                context={
                    "validation": {
                        "parameter": "gpu_memory_utilization",
                        "invalid_value": self.gpu_memory_utilization,
                        "corrected_value": 0.9
                    }
                }
            )
            self.gpu_memory_utilization = 0.9

        # 최대 모델 ~
        self.max_model_len = max_model_len or config.engine.max_model_len
        if self.max_model_len is not None and self.max_model_len < 1:
            logger.warning(
                f"Invalid max_model_len: {self.max_model_len}",
                context={
                    "validation": {
                        "parameter": "max_model_len",
                        "invalid_value": self.max_model_len
                    }
                }
            )

        # 최대 시퀀스 수
        self.max_num_seqs = max_num_seqs or config.engine.max_num_seqs
        if self.max_num_seqs < 1:
            default_value = 256
            logger.warning(
                f"Invalid max_num_seqs: {self.max_num_seqs}. Setting to {default_value}",
                context={
                    "validation": {
                        "parameter": "max_num_seqs",
                        "invalid_value": self.max_num_seqs,
                        "corrected_value": default_value
                    }
                }
            )
            self.max_num_seqs = default_value

        # 최대 배치 토큰 수
        self.max_num_batched_tokens = max_num_batched_tokens or config.engine.max_num_batched_tokens
        if self.max_num_batched_tokens < 1:
            default_value = 8192
            logger.warning(
                f"Invalid max_num_batched_tokens: {self.max_num_batched_tokens}. Setting to {default_value}",
                context={
                    "validation": {
                        "parameter": "max_num_batched_tokens",
                        "invalid_value": self.max_num_batched_tokens,
                        "corrected_value": default_value
                    }
                }
            )
            self.max_num_batched_tokens = default_value

        # 스왑 공간
        self.swap_space = swap_space or config.engine.swap_space
        if self.swap_space < 0:
            default_value = 4
            logger.warning(
                f"Invalid swap_space: {self.swap_space}. Setting to {default_value}",
                context={
                    "validation": {
                        "parameter": "swap_space",
                        "invalid_value": self.swap_space,
                        "corrected_value": default_value
                    }
                }
            )
            self.swap_space = default_value

        # 원격 코드 신뢰 여부
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else config.model.trust_remote_code

        # 데이터 타입
        valid_dtypes = {"auto", "float16", "bfloat16", "float32"}
        self.dtype = dtype or config.engine.dtype
        if self.dtype not in valid_dtypes:
            default_dtype = "auto"
            logger.warning(
                f"Invalid dtype: {self.dtype}. Setting to '{default_dtype}'",
                context={
                    "validation": {
                        "parameter": "dtype",
                        "invalid_value": self.dtype,
                        "corrected_value": default_dtype,
                        "valid_options": list(valid_dtypes)
                    }
                }
            )
            self.dtype = default_dtype

        # Eager 모드 강제 실행 여부
        self.enforce_eager = enforce_eager if enforce_eager is not None else config.engine.enforce_eager

        # 양자화 설정
        valid_quantization = {None, "awq", "gptq", "squeezellm"}
        if quantization is None and config.quantization.enabled:
            self.quantization = config.quantization.method
        else:
            self.quantization = quantization

        if self.quantization not in valid_quantization:
            logger.warning(
                f"Unsupported quantization: {self.quantization}. Setting to None",
                context={
                    "validation": {
                        "parameter": "quantization",
                        "invalid_value": self.quantization,
                        "corrected_value": None,
                        "valid_options": list(valid_quantization)
                    }
                }
            )
            self.quantization = None

        if self.is_master and not performance_mode:
            logger.debug(
                "Parameter validation completed",
                context=validation_context
            )

    def _setup_gpu(self):
        """GPU 설정 및 확인"""
        if torch.cuda.is_available():
            # 현재 프로세스가 사용할 GPU 지정
            torch.cuda.set_device(self.local_rank)

            # 사용 가능한 GPU 수 확인
            num_gpus = torch.cuda.device_count()

            if self.is_master:
                if performance_mode:
                    logger.info(f"Using distributed mode", context={
                        "distributed": {"processes": self.world_size}
                    })
                else:
                    logger.info(
                        f"Using distributed mode",
                        context={
                            "distributed": {
                                "processes": self.world_size,
                                "gpus_per_node": num_gpus
                            }
                        }
                    )

            # 각 프로세스의 GPU 할당 정보
            gpu_name = torch.cuda.get_device_name(self.local_rank)
            logger.info(
                f"Process using GPU",
                context={
                    "process": {
                        "rank": self.rank,
                        "local_rank": self.local_rank,
                        "gpu_name": gpu_name
                    }
                }
            )

            return num_gpus
        else:
            logger.warning(
                "No GPU available, using CPU. Performance will be significantly degraded.",
                context={"hardware": {"is_cuda_available": False}}
            )
            return 0

    def _compute_tensor_parallel_size(self, tensor_parallel_size):
        """텐서 병렬 크기 계산 및 검증"""
        # 텐서 병렬 크기가 설정되지 않은 경우, 적절한 값 계산
        if tensor_parallel_size is None:
            # 텐서 병렬 크기 = min(노드당 GPU 수, 8)
            tensor_parallel_size = min(self.num_gpus if self.num_gpus > 0 else 1, 8)
            logger.info(
                f"Automatically set tensor parallel size",
                context={
                    "distributed": {
                        "tensor_parallel_size": tensor_parallel_size,
                        "auto_configured": True
                    }
                }
            )

        # 병렬 크기 유효성 검사 수행
        tensor_parallel_size = self._validate_tensor_parallel_size(tensor_parallel_size)
        return tensor_parallel_size

    def _validate_tensor_parallel_size(self, tensor_parallel_size):
        """텐서 병렬 크기 유효성 검사"""
        validation_context = {
            "validation": {
                "parameter": "tensor_parallel_size",
                "original_value": tensor_parallel_size
            }
        }

        # 텐서 병렬 크기가 정수인지 확인
        try:
            tensor_parallel_size = int(tensor_parallel_size)
            validation_context["validation"]["parsed_value"] = tensor_parallel_size
        except (ValueError, TypeError):
            logger.warning(
                f"Tensor parallel size must be an integer",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "invalid_value": tensor_parallel_size,
                        "corrected_value": 1,
                        "reason": "not_an_integer"
                    }
                }
            )
            return 1

        # 텐서 병렬 크기가 1보다 작은 경우
        if tensor_parallel_size < 1:
            logger.warning(
                f"Tensor parallel size must be at least 1",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "invalid_value": tensor_parallel_size,
                        "corrected_value": 1,
                        "reason": "less_than_one"
                    }
                }
            )
            return 1

        # 텐서 병렬 크기가 GPU 수보다 큰 경우
        if 0 < self.num_gpus < tensor_parallel_size:
            logger.warning(
                f"Requested tensor parallel size exceeds available GPUs",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "requested_value": tensor_parallel_size,
                        "available_gpus": self.num_gpus,
                        "corrected_value": self.num_gpus,
                        "reason": "exceeds_available_gpus"
                    }
                }
            )
            return self.num_gpus

        # 텐서 병렬 크기가 2의 거듭제곱인지 확인
        if tensor_parallel_size > 1 and (tensor_parallel_size & (tensor_parallel_size - 1)) != 0:
            # 가장 가까운 2의 거듭제곱 값 찾기
            next_power = 2 ** (tensor_parallel_size.bit_length() - 1)
            if next_power * 2 <= self.num_gpus:
                next_power *= 2

            logger.warning(
                f"Tensor parallel size should be a power of 2 for optimal performance",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "requested_value": tensor_parallel_size,
                        "corrected_value": next_power,
                        "reason": "not_power_of_two"
                    }
                }
            )
            return next_power

        # 텐서 병렬 크기가 너무 큰 경우 (일반적으로 16 이상은 성능 저하 가능성)
        if tensor_parallel_size > 16:
            logger.warning(
                f"Very large tensor parallel size may cause performance degradation",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "value": tensor_parallel_size,
                        "recommended_max": 16,
                        "concern": "performance_overhead"
                    }
                }
            )

        # 텐서 병렬 크기가 세계 크기(world_size)에 나누어 떨어지는지 확인
        if self.world_size % tensor_parallel_size != 0:
            logger.warning(
                f"Tensor parallel size is not a divisor of world size",
                context={
                    "validation": {
                        "parameter": "tensor_parallel_size",
                        "value": tensor_parallel_size,
                        "world_size": self.world_size,
                        "concern": "uneven_distribution"
                    }
                }
            )

        # 모든 검증을 통과한 경우
        logger.info(
            f"Validated tensor parallel size",
            context={
                "distributed": {
                    "tensor_parallel_size": tensor_parallel_size,
                    "validation": "passed"
                }
            }
        )
        return tensor_parallel_size

    def _create_engine_args(self):
        """vLLM 엔진 인자 생성"""
        engine_args = AsyncEngineArgs(
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

        logger.debug(
            "Created engine arguments",
            context={
                "engine_creation": {
                    "model": self.model_name,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "pipeline_parallel_size": self.pipeline_parallel_size
                }
            }
        )

        return engine_args

    def _log_initialization_params(self):
        """초기화 매개변수 로깅"""
        # 상세 설정 정보를 구조화된 형태로 로깅
        init_params = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "quantization": self.quantization,
            "swap_space": self.swap_space,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "block_size": config.engine.block_size
        }

        logger.info(
            "Distributed engine configuration",
            context={"initialization": {"parameters": init_params}}
        )

    def _init_distributed(self):
        """분산 환경 초기화"""
        # 이미 초기화된 경우 스킵
        if dist.is_initialized():
            logger.info("Distributed environment is already initialized")
            return

        # 분산 환경 변수 설정
        env_vars = {
            "MASTER_ADDR": self.dist_config.master_addr,
            "MASTER_PORT": self.dist_config.master_port,
            "WORLD_SIZE": str(self.dist_config.world_size),
            "RANK": str(self.dist_config.rank),
            "LOCAL_RANK": str(self.dist_config.local_rank)
        }

        # 환경 변수 설정 로깅
        logger.debug(
            "Setting distributed environment variables",
            context={"distributed": {"env_vars": env_vars}}
        )

        # 환경 변수 적용
        for key, value in env_vars.items():
            os.environ[key] = value

        # 분산 환경 초기화 시도
        with TimingContext(logger, "Distributed initialization") as timing:
            try:
                init_context = {
                    "distributed": {
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "backend": self.dist_config.backend,
                        "init_method": self._get_init_method(),
                        "timeout_seconds": self.dist_config.timeout
                    }
                }

                logger.info(
                    "Initializing distributed environment",
                    context=init_context
                )

                with profile_block("distributed_init"):
                    dist.init_process_group(
                        backend=self.dist_config.backend,
                        init_method=self._get_init_method(),
                        world_size=self.world_size,
                        rank=self.rank,
                        timeout=datetime.timedelta(seconds=self.dist_config.timeout)
                    )

                logger.info(
                    "Distributed environment initialized successfully",
                    context={
                        "distributed": {
                            "initialization": {
                                "status": "success",
                                "duration_seconds": timing.duration
                            }
                        }
                    }
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize distributed environment",
                    context={
                        "distributed": {
                            "initialization": {
                                "status": "failed",
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        }
                    },
                    exc_info=True
                )
                raise

    def _get_init_method(self) -> str:
        """분산 초기화 방법 문자열 생성"""
        if self.dist_config.init_method:
            logger.debug(
                "Using explicit init_method",
                context={"distributed": {"init_method": self.dist_config.init_method}}
            )
            return self.dist_config.init_method

        # 기본값: TCP 초기화 방법
        init_method = f"tcp://{self.dist_config.master_addr}:{self.dist_config.master_port}"
        logger.debug(
            "Using TCP init_method",
            context={"distributed": {"init_method": init_method}}
        )
        return init_method

    def _sync_distributed(self):
        """분산 환경 동기화"""
        if dist.is_initialized():
            # 마스터 노드이고 성능 모드가 아닌 경우만 상세 로깅
            should_log = self.is_master and not performance_mode

            if should_log:
                logger.info("Synchronizing distributed processes...")

            with TimingContext(logger if should_log else None, "Distributed synchronization") as timing:
                try:
                    dist.barrier()
                    if should_log:
                        logger.info(
                            "Distributed processes synchronized",
                            context={
                                "distributed": {
                                    "synchronization": {
                                        "status": "success",
                                        "duration_seconds": timing.duration
                                    }
                                }
                            }
                        )
                except Exception as e:
                    # 오류는 항상 로깅
                    logger.error(
                        "Error during distributed synchronization",
                        context={
                            "distributed": {
                                "synchronization": {
                                    "status": "failed",
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)
                                }
                            }
                        },
                        exc_info=True
                    )
                    raise

    @profile
    @with_request_context
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

        # 요청 컨텍스트 설정 - 구조화된 로깅 활용
        logger.set_request_id(request_id)
        logger.with_context(
            distributed=True,
            is_master=self.is_master,
            rank=self.rank
        )

        # 요청 정보 로깅 - 마스터 노드에서만 INFO 수준으로, 성능 모드 고려
        log_method = logger.info if (self.is_master and not performance_mode) else logger.debug
        log_method(
            "Request received",
            context={
                "request": {
                    "id": request_id,
                    "prompt_length": len(req_config.prompt),
                    "stream": req_config.stream
                }
            }
        )

        # 메트릭 수집기 가져오기 (마스터 노드에서만 메트릭 수집)
        metrics_collector = None
        if config.monitoring.enabled:
            metrics_collector = get_metrics_collector() if self.is_master else None
            # 마스터 노드의 경우 메트릭 수집 시작
            if self.is_master and metrics_collector:
                metrics_collector.start_request(request_id)

        # 스레드 안전을 위한 락 획득 (요청 카운터 업데이트 시)
        async with self._request_lock:
            # 활성 요청 수 증가
            self.active_requests += 1
            self.total_requests += 1
            self.request_metrics[request_id]["status"] = "processing"

        # 타이밍 컨텍스트 시작
        timing = TimingContext(logger, "Request processing")

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
                log_method(
                    "Starting streaming response",
                    context={"request": {"mode": "streaming"}}
                )
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
                error_msg = "No result generated for request"
                logger.error(
                    error_msg,
                    context={"request": {"status": "failed", "reason": "empty_result"}}
                )
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
            if config.monitoring.enabled:
                if self.is_master and metrics_collector:
                    metrics_collector.complete_request(
                        request_id,
                        completion_tokens=completion_tokens,
                        prompt_tokens=prompt_tokens
                    )

            # 통계 로깅 - 구조화된 형식으로
            if self.is_master:
                log_method(
                    "Request completed",
                    context={
                        "performance": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "duration_seconds": timing.duration,
                            "tokens_per_second": tokens_per_second
                        }
                    }
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
            logger.error(
                "Error processing request",
                context={
                    "request": {
                        "id": request_id,
                        "status": "failed"
                    },
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)
                    }
                },
                exc_info=True
            )
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
                    if remove_count > 0:
                        logger.debug(
                            "Cleaning up old request metrics",
                            context={
                                "cleanup": {
                                    "total_metrics": len(self.request_metrics),
                                    "removing_count": remove_count
                                }
                            }
                        )
                    for req_id in sorted(completed_requests,
                                         key=lambda x: self.request_metrics[x].get("end_time", 0))[:remove_count]:
                        del self.request_metrics[req_id]

            # 요청 컨텍스트 정리
            logger.clear_context()
            logger.set_request_id(None)

    @classmethod
    def _validate_request_config(cls, req_config: RequestConfig):
        """요청 설정 검증"""

        if not req_config.prompt:
            error_msg = "Prompt is empty"
            logger.error(
                error_msg,
                context={
                    "validation": {
                        "parameter": "prompt",
                        "status": "failed",
                        "reason": "empty_prompt"
                    }
                }
            )
            raise ValueError(error_msg)

        if req_config.max_tokens is not None and req_config.max_tokens <= 0:
            logger.warning(
                f"Invalid max_tokens: {req_config.max_tokens}. Using default value.",
                context={
                    "validation": {
                        "parameter": "max_tokens",
                        "invalid_value": req_config.max_tokens,
                        "action": "using_default"
                    }
                }
            )
            req_config.max_tokens = config.inference.max_tokens

        if req_config.temperature is not None and req_config.temperature < 0:
            logger.warning(
                f"Invalid temperature: {req_config.temperature}. Using default value.",
                context={
                    "validation": {
                        "parameter": "temperature",
                        "invalid_value": req_config.temperature,
                        "action": "using_default"
                    }
                }
            )
            req_config.temperature = config.inference.temperature

    @classmethod
    def _create_engine_request(cls, req_config: RequestConfig) -> RequestConfig:
        """엔진 요청 객체 생성"""
        logger.debug(
            "Creating engine request",
            context={
                "request": {
                    "id": req_config.request_id,
                    "max_tokens": req_config.max_tokens,
                    "temperature": req_config.temperature
                }
            }
        )

        return RequestConfig(
            prompt=req_config.prompt,
            max_tokens=req_config.max_tokens,
            temperature=req_config.temperature,
            top_p=req_config.top_p,
            top_k=req_config.top_k,
            frequency_penalty=req_config.frequency_penalty,
            presence_penalty=req_config.presence_penalty,
            repetition_penalty=req_config.repetition_penalty,
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
        스트리밍 생성 요청 처리를 위한 제너레이터 (배치 처리 적용, 네트워크 효율성 향상)

        Args:
            engine_request: 요청 객체
            sampling_params: SamplingParams 객체
            timing: 타이밍 컨텍스트
            metrics_collector: 메트릭 수집기 (없으면 None)

        Yields:
            생성된 텍스트 조각
        """
        request_id = engine_request.request_id
        stream_finished = False
        previous_text_length = 0  # 이전 텍스트 길이 추적용 변수
        is_first_response = True  # 첫 번째 응답 여부 체크

        # 스트리밍 컨텍스트 추가
        logger.with_context(streaming=True)

        # 배치 처리 관련 변수
        batch_size_chars = 20  # 약 5-10개 토큰에 해당 (한글은 대략 2-3자가 1토큰)
        batch_max_wait_time = 0.1  # 최대 대기 시간 (초)
        current_batch = ""  # 현재 배치에 모인 텍스트
        last_send_time = time.time()  # 마지막 전송 시간

        # 타이밍 시작
        timing.__enter__()

        logger.debug(
            "Stream generation started",
            context={
                "streaming": {
                    "request_id": request_id,
                    "batch_size_chars": batch_size_chars,
                    "batch_max_wait_time": batch_max_wait_time
                }
            }
        )

        try:
            async for result in self.engine.generate(
                    prompt=engine_request.prompt,
                    sampling_params=sampling_params,
                    request_id=request_id):

                output = result.outputs[0]
                tokens_generated = len(output.token_ids)
                prompt_tokens = len(result.prompt_token_ids)

                # 새로 생성된 텍스트 부분 계산
                current_text = output.text
                new_text = current_text[previous_text_length:]
                previous_text_length = len(current_text)

                # 디버그 로깅 - 상세하지만 TRACE 수준 정보는 조건부로
                if config.logging.level.upper() == "DEBUG" and tokens_generated % 10 == 0:  # 10토큰마다 로깅
                    logger.debug(
                        "Stream progress",
                        context={
                            "streaming": {
                                "tokens_generated": tokens_generated,
                                "new_text_length": len(new_text)
                            }
                        }
                    )

                # 새 텍스트를 현재 배치에 추가
                current_batch += new_text
                current_time = time.time()
                time_since_last_send = current_time - last_send_time

                # 배치 전송 조건 확인
                is_finished = output.finish_reason is not None
                should_send_batch = (
                        len(current_batch) >= batch_size_chars or
                        time_since_last_send >= batch_max_wait_time or
                        is_finished
                )

                if should_send_batch and current_batch:  # 배치에 내용이 있을 경우에만 전송
                    # 응답 객체 구성
                    response = {
                        "id": request_id,
                        "new_text": current_batch,
                        "finished": is_finished,
                        "finish_reason": output.finish_reason
                    }

                    # 전체 텍스트는 첫 번째 응답과 마지막 응답에만 포함
                    if is_first_response or is_finished:
                        response["generated_text"] = current_text
                        is_first_response = False

                    # 배치 전송 로깅 - TRACE 수준으로 간주하고 DEBUG에서만 상세 정보
                    if not performance_mode and config.logging.level.upper() == "DEBUG":
                        logger.debug(f"Stream chunk received", context={
                            "streaming": {
                                "new_text_length": len(new_text),
                                "total_text_length": len(current_text)
                            }
                        })

                    # 배치 전송
                    yield response

                    # 배치 초기화 및 전송 시간 업데이트
                    current_batch = ""
                    last_send_time = current_time

                # 마지막 출력인 경우 통계 업데이트 (스트림이 완료되었고 아직 처리되지 않은 경우)
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

                    # 로깅 - 구조화된 형식
                    tokens_per_second = tokens_generated / timing.duration if timing.duration > 0 else 0
                    logger.info(
                        "Streaming request completed",
                        context={
                            "performance": {
                                "tokens_generated": tokens_generated,
                                "prompt_tokens": prompt_tokens,
                                "duration_seconds": timing.duration,
                                "tokens_per_second": tokens_per_second
                            }
                        }
                    )

        except Exception as e:
            logger.error(
                "Error in stream generation",
                context={
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "request_id": request_id
                    }
                },
                exc_info=True
            )

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
            async with self._request_lock:
                if self.request_metrics[request_id]["status"] == "processing":
                    self.active_requests -= 1

            # 스트리밍 컨텍스트 정리
            logger.with_context(streaming=False)
