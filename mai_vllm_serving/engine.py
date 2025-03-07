"""
mai-vllm-serving의 엔진 구현
vLLM 엔진의 성능과 안정성을 최적화하기 위한 래퍼 클래스
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, AsyncIterator

import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput

from mai_vllm_serving.monitoring.metrics import get_metrics_collector
from mai_vllm_serving.monitoring.profiler import profile, profile_block, get_profiler
# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config
# 로깅 유틸리티 임포트
from mai_vllm_serving.utils.logging_utils import (
    setup_logging,
    TimingContext
)

# 설정 객체 가져오기
config = get_config()

# 로깅 초기화
logger = setup_logging(
    service_name="mai-vllm-serving-engine",
    log_level=config.logging.level,
    use_json=config.logging.json,
    log_file=config.logging.file
)


@dataclass
class RequestConfig:
    """추론 요청 설정을 위한 데이터 클래스"""
    prompt: str
    max_tokens: int = config.inference.max_tokens
    temperature: float = config.inference.temperature
    top_p: float = config.inference.top_p
    top_k: int = config.inference.top_k
    frequency_penalty: float = config.inference.frequency_penalty
    presence_penalty: float = config.inference.presence_penalty
    repetition_penalty: float = config.inference.repetition_penalty
    # no_repeat_ngram_size: int = config.inference.no_repeat_ngram_size
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    request_id: Optional[str] = None
    seed: Optional[int] = config.inference.seed

    def to_sampling_params(self) -> SamplingParams:
        """
        RequestConfig 객체를 SamplingParams 객체로 변환

        Returns:
            SamplingParams: vLLM에서 사용할 샘플링 파라미터
        """
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
            seed=self.seed,
            ignore_eos=False
            # no_repeat_ngram_size=self.no_repeat_ngram_size
        )


@dataclass
class InferenceStats:
    """추론 통계 정보를 위한 데이터 클래스"""
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    start_time: float
    end_time: float = field(default=0.0)

    @property
    def inference_time(self) -> float:
        """추론 시간(초)"""
        if self.end_time == 0.0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """초당 생성 토큰 수"""
        if self.inference_time == 0.0 or self.completion_tokens == 0:
            return 0.0
        return self.completion_tokens / self.inference_time


class MAIVLLMEngine:
    """
    mai-vllm-serving 엔진

    vLLM 엔진의 성능과 안정성을 최적화하기 위한 래퍼 클래스
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 tensor_parallel_size: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 max_model_len: Optional[int] = None,
                 max_num_seqs: Optional[int] = None,
                 max_num_batched_tokens: Optional[int] = None,
                 quantization: Optional[str] = None,
                 swap_space: Optional[int] = None,
                 trust_remote_code: Optional[bool] = None,
                 dtype: Optional[str] = None,
                 disable_log_stats: Optional[bool] = None,
                 enforce_eager: Optional[bool] = None):
        """
        vLLM 엔진 초기화

        Args:
            model_name: 허깅페이스 모델 ID 또는 로컬 경로
            tensor_parallel_size: 텐서 병렬 처리에 사용할 GPU 수
            gpu_memory_utilization: GPU 메모리 사용률 (0.0 ~ 1.0)
            max_model_len: 최대 모델 ~
            max_num_seqs: 최대 배치 시퀀스 수
            max_num_batched_tokens: 배치당 최대 토큰 수
            quantization: 양자화 방식 (None, "awq", "gptq", "squeezellm")
            swap_space: CPU 스왑 공간 크기 (GB)
            trust_remote_code: 원격 코드 신뢰 여부
            dtype: 연산 데이터 타입 ("auto", "float16", "bfloat16", "float32")
            disable_log_stats: 통계 로깅 비활성화 여부
            enforce_eager: PyTorch eager 모드 강제 실행 여부
        """
        logger.info("Initializing MAIVLLMEngine")

        # 설정에서 기본값 가져오기
        self.model_name = model_name or config.model.name
        self.start_time = time.time()
        self.inference_stats = {}  # request_id -> InferenceStats
        self.active_requests = 0
        self.total_requests = 0
        self.total_tokens_generated = 0

        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Found {self.num_gpus} GPUs")

            # GPU 정보 로깅
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.2f} GB")
        else:
            self.num_gpus = 0
            logger.warning("No GPUs available, using CPU. Performance will be significantly degraded.")

        # 인자가 None인 경우 설정에서 가져오기
        if tensor_parallel_size is None:
            tensor_parallel_size = config.engine.tensor_parallel_size
        if gpu_memory_utilization is None:
            gpu_memory_utilization = config.engine.gpu_memory_utilization
        if max_model_len is None:
            max_model_len = config.engine.max_model_len
        if max_num_seqs is None:
            max_num_seqs = config.engine.max_num_seqs
        if max_num_batched_tokens is None:
            max_num_batched_tokens = config.engine.max_num_batched_tokens
        if swap_space is None:
            swap_space = config.engine.swap_space
        if trust_remote_code is None:
            trust_remote_code = config.model.trust_remote_code
        if dtype is None:
            dtype = config.engine.dtype
        if disable_log_stats is None:
            disable_log_stats = config.engine.disable_log_stats
        if enforce_eager is None:
            enforce_eager = config.engine.enforce_eager

        # 양자화 설정
        if quantization is None and config.quantization.enabled:
            quantization = config.quantization.method
            logger.info(f"Using quantization method: {quantization}")

        # tensor_parallel_size 조정
        if tensor_parallel_size is None:
            # 설정에 값이 없는 경우 사용 가능한 GPU 수로 설정
            tensor_parallel_size = max(1, self.num_gpus)
            logger.info(f"Setting tensor_parallel_size to {tensor_parallel_size}")

        # tensor_parallel_size가 사용 가능한 GPU 수를 초과하는 경우 경고
        if 0 < self.num_gpus < tensor_parallel_size:
            logger.warning(
                f"Requested tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({self.num_gpus})")
            logger.warning(f"Setting tensor_parallel_size to {self.num_gpus}")
            tensor_parallel_size = self.num_gpus

        # 엔진 설정 로깅
        logger.info(f"Initializing vLLM engine with model: {self.model_name}")
        logger.info(f"Engine configuration:")
        logger.info(f"  - Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  - GPU memory utilization: {gpu_memory_utilization}")
        logger.info(f"  - Max num sequences: {max_num_seqs}")
        logger.info(f"  - Max num batched tokens: {max_num_batched_tokens}")
        logger.info(f"  - Block size: {config.engine.block_size}")
        logger.info(f"  - Swap space: {swap_space} GB")
        logger.info(f"  - Data type: {dtype}")

        # 엔진 인자 설정
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            quantization=quantization,
            swap_space=swap_space,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            disable_log_stats=disable_log_stats,
            enforce_eager=enforce_eager,
            block_size=config.engine.block_size
        )

        # 엔진 초기화
        with TimingContext(logger, "Engine initialization") as timing:
            try:
                with profile_block("engine_initialization"):  # 프로파일링 블록 추가
                    self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info(f"Engine initialization completed successfully in {timing.duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to initialize engine: {str(e)}", exc_info=True)
                raise

    @profile
    async def generate(self, req_config: RequestConfig) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        텍스트 생성 요청 처리

        Args:
            req_config: 생성 요청 설정

        Returns:
            생성된 텍스트와 관련 메타데이터 또는 스트리밍 응답의 AsyncIterator
        """
        # 요청 ID 생성
        if req_config.request_id is None:
            req_config.request_id = str(uuid.uuid4())

        # 메트릭 수집기 가져오기
        metrics_collector = get_metrics_collector()

        # 요청 처리 시작 기록 (프롬프트 토큰 수는 아직 모름)
        metrics_collector.start_request(req_config.request_id)

        # 추론 시작 로깅
        logger.info(f"Request {req_config.request_id} received: prompt length={len(req_config.prompt)}, "
                    f"max_tokens={req_config.max_tokens}, temperature={req_config.temperature}")

        # 타이밍 컨텍스트 시작
        with TimingContext(logger, f"Request {req_config.request_id} processing") as timing:
            # 요청 통계 초기화
            self.inference_stats[req_config.request_id] = InferenceStats(
                request_id=req_config.request_id,
                prompt_tokens=0,  # 나중에 업데이트
                completion_tokens=0,  # 나중에 업데이트
                start_time=time.time()
            )

            # 활성 요청 수 증가
            self.active_requests += 1
            self.total_requests += 1

            try:
                with timing:
                    # engine_request 생성 시 요청 값 우선 사용, 없을 경우에만 기본값 적용
                    engine_request = RequestConfig(
                        prompt=req_config.prompt,
                        # 아래 값들은 요청에 있을 경우 그 값을 사용, 없으면 config의 기본값 사용
                        max_tokens=req_config.max_tokens if req_config.max_tokens is not None else config.inference.max_tokens,
                        temperature=req_config.temperature if req_config.temperature is not None else config.inference.temperature,
                        top_p=req_config.top_p if req_config.top_p is not None else config.inference.top_p,
                        top_k=req_config.top_k if req_config.top_k is not None else config.inference.top_k,
                        frequency_penalty=req_config.frequency_penalty if req_config.frequency_penalty is not None else config.inference.frequency_penalty,
                        presence_penalty=req_config.presence_penalty if req_config.presence_penalty is not None else config.inference.presence_penalty,
                        repetition_penalty=req_config.repetition_penalty if req_config.repetition_penalty is not None else config.inference.repetition_penalty,
                        # no_repeat_ngram_size=req_config.no_repeat_ngram_size if req_config.no_repeat_ngram_size is not None else config.inference.no_repeat_ngram_size,
                        stop=req_config.stop,
                        stream=req_config.stream,
                        request_id=req_config.request_id,
                        seed=config.inference.seed
                    )

                # SamplingParams 생성
                sampling_params = engine_request.to_sampling_params()

                # 스트리밍 모드인 경우
                if req_config.stream:
                    logger.debug(f"Starting streaming response for request {req_config.request_id}")
                    with profile_block(f"stream_request_{req_config.request_id}"):
                        return self._stream_generator(engine_request, sampling_params, timing, metrics_collector)

                # 일반 모드인 경우
                result = None
                with profile_block(f"generate_request_{engine_request.request_id}"):
                    async for res in self.engine.generate(
                            prompt=engine_request.prompt,
                            sampling_params=sampling_params,
                            request_id=engine_request.request_id):
                        result = res

                if result is None:
                    err_msg = f"No result generated for request {req_config.request_id}"
                    logger.error(err_msg)
                    metrics_collector.fail_request(req_config.request_id, err_msg)
                    raise RuntimeError(err_msg)

                # 결과 파싱 및 통계 업데이트
                output = result.outputs[0]
                stats = self._update_stats(req_config.request_id, result, timing)

                # 메트릭 수집기에 완료 기록
                metrics_collector.complete_request(
                    req_config.request_id,
                    completion_tokens=len(output.token_ids),
                    prompt_tokens=len(result.prompt_token_ids)
                )

                # 로깅
                logger.info(
                    f"Request {req_config.request_id} completed: {stats.prompt_tokens} prompt tokens, "
                    f"{stats.completion_tokens} completion tokens in {stats.inference_time:.3f}s "
                    f"({stats.tokens_per_second:.2f} tokens/sec)"
                )

                return {
                    "id": req_config.request_id,
                    "generated_text": output.text,
                    "usage": {
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "total_tokens": stats.prompt_tokens + stats.completion_tokens
                    },
                    "performance": {
                        "inference_time": f"{stats.inference_time:.3f}",
                        "tokens_per_second": f"{stats.tokens_per_second:.2f}"
                    },
                    "finish_reason": output.finish_reason
                }

            except Exception as e:
                logger.error(f"Error processing request {req_config.request_id}: {str(e)}", exc_info=True)
                # 메트릭 수집기에 실패 기록
                metrics_collector.fail_request(req_config.request_id, str(e))
                raise

            finally:
                # 활성 요청 수 감소
                self.active_requests -= 1

    async def _stream_generator(self, engine_request: RequestConfig,
                                sampling_params: SamplingParams,
                                timing: TimingContext,
                                metrics_collector=None) -> AsyncIterator[Dict[str, Any]]:
        """
        스트리밍 생성 요청 처리를 위한 제너레이터

        Args:
            engine_request: 요청 객체
            sampling_params: SamplingParams 객체
            timing: 타이밍 컨텍스트
            metrics_collector: 메트릭 수집기 (없으면 가져옴)

        Yields:
            생성된 텍스트 조각
        """
        if metrics_collector is None:
            metrics_collector = get_metrics_collector()

        try:
            # last_output = None
            # tokens_generated = 0
            # prompt_tokens = 0

            async for result in self.engine.generate(
                    prompt=engine_request.prompt,
                    sampling_params=sampling_params,
                    request_id=engine_request.request_id):
                output = result.outputs[0]
                tokens_generated = len(output.token_ids)
                prompt_tokens = len(result.prompt_token_ids)

                # 마지막 출력인 경우 통계 업데이트
                if output.finish_reason is not None:
                    self._update_stats(engine_request.request_id, result, timing)

                    # 메트릭 수집기에 완료 기록
                    metrics_collector.complete_request(
                        engine_request.request_id,
                        completion_tokens=tokens_generated,
                        prompt_tokens=prompt_tokens
                    )

                    logger.info(
                        f"Streaming request {engine_request.request_id} completed: {tokens_generated} tokens in "
                        f"{timing.duration:.3f}s ({tokens_generated / timing.duration if timing.duration > 0 else 0:.2f} tokens/sec)"
                    )

                # last_output = output

                yield {
                    "id": engine_request.request_id,
                    "generated_text": output.text,
                    "finished": output.finish_reason is not None,
                    "finish_reason": output.finish_reason
                }

        except Exception as e:
            logger.error(f"Error in stream generation for request {engine_request.request_id}: {str(e)}", exc_info=True)
            # 메트릭 수집기에 실패 기록
            metrics_collector.fail_request(engine_request.request_id, str(e))
            yield {
                "id": engine_request.request_id,
                "error": str(e),
                "finished": True
            }

        finally:
            # 활성 요청 수 감소
            self.active_requests -= 1

    def _update_stats(self, request_id: str, result: RequestOutput, timing: TimingContext) -> InferenceStats:
        """
        요청 통계 업데이트

        Args:
            request_id: 요청 ID
            result: 생성 결과
            timing: 타이밍 컨텍스트

        Returns:
            업데이트 된 추론 통계
        """
        if request_id not in self.inference_stats:
            logger.warning(f"Request ID {request_id} not found in inference_stats")
            return InferenceStats(
                request_id=request_id,
                prompt_tokens=0,
                completion_tokens=0,
                start_time=0.0,
                end_time=0.0
            )

        stats = self.inference_stats[request_id]
        stats.prompt_tokens = len(result.prompt_token_ids)
        stats.completion_tokens = len(result.outputs[0].token_ids)
        stats.end_time = time.time()

        # 전체 생성 토큰 수 업데이트
        self.total_tokens_generated += stats.completion_tokens

        # 타이밍 정보 업데이트 (타이밍 컨텍스트가 더 정확함)
        if timing and timing.duration > 0:
            # 타이밍 컨텍스트의 시간을 사용
            stats.start_time = timing.start_time
            stats.end_time = timing.end_time if timing.end_time else time.time()

        # 통계 로깅 (디버그 레벨로 상세 정보 로깅)
        logger.debug(
            f"Request {request_id} stats: {stats.prompt_tokens} prompt tokens, "
            f"{stats.completion_tokens} completion tokens, {stats.inference_time:.3f}s, "
            f"{stats.tokens_per_second:.2f} tokens/sec"
        )

        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """
        엔진 성능 통계 조회

        Returns:
            성능 통계 정보
        """
        # GPU 사용량 정보
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                gpu_mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                gpu_info.append({
                    "id": i,
                    "name": gpu_name,
                    "memory_allocated_gb": f"{gpu_mem_allocated:.2f}",
                    "memory_reserved_gb": f"{gpu_mem_reserved:.2f}"
                })

        # 평균 처리 속도 계산
        uptime = time.time() - self.start_time
        tokens_per_second = self.total_tokens_generated / uptime if uptime > 0 else 0

        # 메트릭 수집기에서 추가 정보 가져오기
        try:
            metrics_collector = get_metrics_collector()
            metrics_info = metrics_collector.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get metrics from collector: {str(e)}")
            metrics_info = {}

        # 기본 통계 정보
        stats = {
            "model": self.model_name,
            "uptime": f"{uptime:.2f}",
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_second_avg": f"{tokens_per_second:.2f}",
            "gpu_info": gpu_info
        }

        # 메트릭 정보가 있으면 추가
        if metrics_info:
            stats["detailed_metrics"] = {
                "total_requests": metrics_info.get("total_requests", 0),
                "total_successes": metrics_info.get("total_successes", 0),
                "total_failures": metrics_info.get("total_failures", 0),
                "total_tokens_input": metrics_info.get("total_tokens_input", 0),
                "total_tokens_output": metrics_info.get("total_tokens_output", 0),
                "avg_tokens_per_second": metrics_info.get("avg_tokens_per_second", 0),
                "avg_request_time": metrics_info.get("avg_request_time", 0)
            }

        # 프로파일링 정보 추가
        try:
            profiler = get_profiler()
            memory_report = profiler.get_memory_report()
            function_stats = profiler.get_function_stats(top_n=5)  # 상위 5개 함수만

            memory_data = await memory_report
            function_data = await function_stats

            stats["profiling"] = {
                "memory": {
                    "process_ram_gb": await memory_data["current"]["process_ram_used_gb"],
                    "system_ram_percent": await memory_data["current"]["system_ram_percent"]
                },
                "top_functions": [
                    {
                        "name": f"{func['module_name']}.{func['function_name']}",
                        "avg_time": func["avg_time"],
                        "call_count": func["call_count"]
                    } for func in function_data.get("top_functions", [])[:5]
                ]
            }
        except Exception as e:
            logger.warning(f"Failed to get profiling stats: {str(e)}")

        return stats


# 테스트용 코드
async def main():
    """메인 함수 (테스트용)"""
    # 로깅 설정 업데이트
    logging_level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(level=logging_level, format=config.logging.format)

    # 엔진 인스턴스 생성
    engine = MAIVLLMEngine()

    # 샘플 요청 생성
    test_config = RequestConfig(
        prompt="Hello, how are you?",
        max_tokens=100
    )

    # 추론 실행
    result = await engine.generate(test_config)
    print(json.dumps(result, indent=2))

    # 성능 통계 출력
    stats = await engine.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import asyncio
    import json

    asyncio.run(main())
