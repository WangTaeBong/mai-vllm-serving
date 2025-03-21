"""
mai-vllm-serving의 엔진 구현
vLLM 엔진의 성능과 안정성을 최적화하기 위한 래퍼 클래스
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, AsyncIterator, AsyncGenerator

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
    get_logger,
    TimingContext,
    with_request_context
)

# 설정 객체 가져오기
config = get_config()

# 성능 모드 여부 확인
performance_mode = config.logging.log_performance_mode

# 구조화된 로거 초기화로 변경
logger = get_logger("engine", production_mode=performance_mode)


# 따옴표 제거 도우미 함수
def remove_quotes_if_needed(text):
    if isinstance(text, str) and text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


@dataclass
class RequestConfig:
    """추론 요청 설정을 위한 데이터 클래스"""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    request_id: Optional[str] = None
    seed: Optional[int] = None

    def to_sampling_params(self) -> SamplingParams:
        """
        RequestConfig 객체를 SamplingParams 객체로 변환

        Returns:
            SamplingParams: vLLM에서 사용할 샘플링 파라미터
        """
        # 환경 설정에서 기본값 가져오기
        cfg = get_config()

        return SamplingParams(
            max_tokens=self.max_tokens if self.max_tokens is not None else cfg.inference.max_tokens,
            temperature=self.temperature if self.temperature is not None else cfg.inference.temperature,
            top_p=self.top_p if self.top_p is not None else cfg.inference.top_p,
            top_k=self.top_k if self.top_k is not None else cfg.inference.top_k,
            frequency_penalty=self.frequency_penalty if self.frequency_penalty is not None else cfg.inference.frequency_penalty,
            presence_penalty=self.presence_penalty if self.presence_penalty is not None else cfg.inference.presence_penalty,
            repetition_penalty=self.repetition_penalty if self.repetition_penalty is not None else cfg.inference.repetition_penalty,
            stop=self.stop,
            seed=self.seed if self.seed is not None else cfg.inference.seed,
            ignore_eos=False
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
        # 구조화된 로깅 사용으로 더 많은 컨텍스트 포함
        logger.info("Initializing MAIVLLMEngine", context={
            "initialization": {
                "component": "engine",
                "version": "0.1.0"
            }
        })

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
            logger.info(f"Found {self.num_gpus} GPUs", context={
                "hardware": {
                    "gpu_count": self.num_gpus,
                    "is_cuda_available": True
                }
            })

            # GPU 정보 로깅 - 세부 정보는 DEBUG 레벨로 변경
            gpu_info = []
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_info.append({
                    "id": i,
                    "name": gpu_name,
                    "memory_gb": f"{gpu_mem:.2f}"
                })

            # 모든 GPU 정보를 한 번에 로깅
            if not performance_mode:
                logger.debug("GPU details", context={"hardware": {"gpus": gpu_info}})
        else:
            self.num_gpus = 0
            logger.warning(
                "No GPUs available, using CPU. Performance will be significantly degraded.",
                context={"hardware": {"is_cuda_available": False}}
            )

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
            logger.info(f"Using quantization method: {quantization}", context={
                "engine": {"quantization": {"method": quantization}}
            })

        # tensor_parallel_size 조정
        if tensor_parallel_size is None:
            # 설정에 값이 없는 경우 사용 가능한 GPU 수로 설정
            tensor_parallel_size = max(1, self.num_gpus)
            logger.info(f"Setting tensor_parallel_size to {tensor_parallel_size}")

        # tensor_parallel_size가 사용 가능한 GPU 수를 초과하는 경우 경고
        if 0 < self.num_gpus < tensor_parallel_size:
            logger.warning(
                f"Requested tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({self.num_gpus})",
                context={
                    "engine": {
                        "requested_tensor_parallel": tensor_parallel_size,
                        "available_gpus": self.num_gpus,
                        "action": "adjusting_down"
                    }
                }
            )
            logger.warning(f"Setting tensor_parallel_size to {self.num_gpus}")
            tensor_parallel_size = self.num_gpus

        # 엔진 설정을 구조화된 방식으로 로깅
        engine_config = {
            "model": self.model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "block_size": config.engine.block_size,
            "swap_space": swap_space,
            "dtype": dtype
        }

        logger.info("Engine configuration", context={"engine": engine_config})

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
        with TimingContext(logger, "Engine initialization", log_threshold=0.1 if performance_mode else None) as timing:
            try:
                with profile_block("engine_initialization"):  # 프로파일링 블록 추가
                    self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                logger.info(
                    f"Engine initialization completed successfully",
                    context={
                        "initialization": {
                            "duration_seconds": timing.duration,
                            "status": "success"
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

    @profile
    @with_request_context
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

        # 메트릭 수집기 가져오기 - 모니터링이 활성화된 경우에만
        metrics_collector = None
        if config.monitoring.enabled:
            metrics_collector = get_metrics_collector()
            # 요청 처리 시작 기록
            metrics_collector.start_request(req_config.request_id)

        # 추론 시작 로깅 - 구조화된 로깅 사용
        request_context = {
            "request": {
                "id": req_config.request_id,
                "prompt_length": len(req_config.prompt),
                "max_tokens": req_config.max_tokens,
                "temperature": req_config.temperature,
                "stream": req_config.stream
            }
        }

        # 요청 ID를 구조화된 로거에 설정
        logger.set_request_id(req_config.request_id)

        logger.info(
            f"Request generate received {req_config.request_id}",
            context=request_context
        )

        # 타이밍 컨텍스트 시작
        with TimingContext(logger, f"Request generate processing") as timing:
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

                # SamplingParams 생성
                sampling_params = engine_request.to_sampling_params()

                # 스트리밍 모드인 경우
                if req_config.stream:
                    logger.debug(f"Starting streaming response", context={
                        "request": {"mode": "streaming"}
                    })
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
                    err_msg = f"No result generated"
                    logger.error(err_msg, context={
                        "request": {"status": "failed", "error": "empty_result"}
                    })
                    # 모니터링이 활성화된 경우에만 메트릭 업데이트
                    if config.monitoring.enabled and metrics_collector:
                        metrics_collector.fail_request(req_config.request_id, err_msg)
                    raise RuntimeError(err_msg)

                # 결과 파싱 및 통계 업데이트
                output = result.outputs[0]
                stats = self._update_stats(req_config.request_id, result, timing)

                # 메트릭 수집기에 완료 기록 - 모니터링이 활성화된 경우에만
                if config.monitoring.enabled and metrics_collector:
                    metrics_collector.complete_request(
                        req_config.request_id,
                        completion_tokens=len(output.token_ids),
                        prompt_tokens=len(result.prompt_token_ids)
                    )

                # 로깅 - 구조화된 형식으로 변경
                if not performance_mode or stats.inference_time > 0.5:  # 500ms 이상 걸린 요청만 로깅
                    logger.info(
                        f"Request completed",
                        context={
                            "performance": {
                                "prompt_tokens": stats.prompt_tokens,
                                "completion_tokens": stats.completion_tokens,
                                "inference_time": stats.inference_time,
                                "tokens_per_second": stats.tokens_per_second
                            }
                        }
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
                logger.error(
                    "Error processing request",
                    context={
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e)
                        }
                    },
                    exc_info=True
                )
                # 메트릭 수집기에 실패 기록 - 모니터링이 활성화된 경우에만
                if config.monitoring.enabled and metrics_collector:
                    metrics_collector.fail_request(req_config.request_id, str(e))
                raise

            finally:
                # 활성 요청 수 감소
                self.active_requests -= 1
                # 요청 ID 해제
                logger.set_request_id(None)

    async def _stream_generator(self, engine_request: RequestConfig,
                                sampling_params: SamplingParams,
                                timing: TimingContext,
                                metrics_collector=None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 생성 요청 처리를 위한 제너레이터 (배치 처리 적용, 네트워크 효율성 향상)

        Args:
            engine_request: 요청 객체
            sampling_params: SamplingParams 객체
            timing: 타이밍 컨텍스트
            metrics_collector: 메트릭 수집기 (없으면 가져옴)

        Yields:
            생성된 텍스트 조각
        """
        # 메트릭 수집기 확인 - 모니터링이 활성화된 경우에만
        if metrics_collector is None and config.monitoring.enabled:
            metrics_collector = get_metrics_collector()

        request_id = engine_request.request_id
        stream_finished = False
        previous_text_length = 0  # 이전 텍스트 길이 추적용 변수
        is_first_response = True  # 첫 번째 응답 여부 체크

        # 배치 처리 관련 변수
        batch_size_chars = 20  # 약 5-10개 토큰에 해당 (한글은 대략 2-3자가 1토큰)
        batch_max_wait_time = 0.1  # 최대 대기 시간 (초)
        current_batch = ""  # 현재 배치에 모인 텍스트
        last_send_time = time.time()  # 마지막 전송 시간

        # 요청 컨텍스트에 스트리밍 정보 추가
        logger.with_context(streaming=True, batch_size=batch_size_chars)

        # 상세 디버그 로깅
        logger.debug("Stream generation started", context={
            "streaming": {
                "batch_size_chars": batch_size_chars,
                "batch_max_wait_time": batch_max_wait_time
            }
        })

        try:
            async for result in self.engine.generate(
                    prompt=engine_request.prompt,
                    sampling_params=sampling_params,
                    request_id=engine_request.request_id):

                output = result.outputs[0]
                tokens_generated = len(output.token_ids)
                prompt_tokens = len(result.prompt_token_ids)

                # 새로 생성된 텍스트 부분 계산
                current_text = output.text
                new_text = current_text[previous_text_length:]
                previous_text_length = len(current_text)

                # 디버그 로깅 - 상세하지만 TRACE 수준 정보는 조건부로
                if not performance_mode and config.logging.level.upper() == "DEBUG" and tokens_generated % 10 == 0:
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
                    # 텍스트 처리
                    current_batch = remove_quotes_if_needed(current_batch)
                    current_text = remove_quotes_if_needed(current_text)

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

                    # 배치 전송
                    yield response

                    # 배치 초기화 및 전송 시간 업데이트
                    current_batch = ""
                    last_send_time = current_time

                # 마지막 출력인 경우 통계 업데이트
                if is_finished and not stream_finished:
                    stream_finished = True

                    # MAIVLLMEngine은 _request_lock을 사용하지 않으므로 직접 _update_stats 호출
                    self._update_stats(engine_request.request_id, result, timing)

                    # 메트릭 수집기에 완료 기록 - 모니터링이 활성화된 경우에만
                    if config.monitoring.enabled and metrics_collector:
                        metrics_collector.complete_request(
                            request_id,
                            completion_tokens=tokens_generated,
                            prompt_tokens=prompt_tokens
                        )

                    # 로깅 - 구조화된 형식으로 변경
                    tokens_per_sec = tokens_generated / timing.duration if timing.duration > 0 else 0
                    logger.info(
                        f"Streaming request completed: {request_id}",
                        context={
                            "streaming": {
                                "tokens_generated": tokens_generated,
                                "duration_seconds": timing.duration,
                                "tokens_per_second": tokens_per_sec
                            }
                        }
                    )

        except Exception as e:
            logger.error(
                "Error in stream generation",
                context={
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)
                    }
                },
                exc_info=True
            )

            # MAIVLLMEngine에는 _request_lock이 없으므로 이 부분 제거
            # 대신 필요한 경우 오류 정보를 inference_stats에 직접 저장
            if request_id in self.inference_stats:
                self.inference_stats[request_id].end_time = time.time()

            # 메트릭 수집기에 실패 기록 - 모니터링이 활성화된 경우에만
            if config.monitoring.enabled and metrics_collector:
                metrics_collector.fail_request(request_id, str(e))

            # 오류 정보 반환
            yield {
                "id": request_id,
                "error": str(e),
                "finished": True
            }

        finally:
            # 활성 요청 수 감소 (MAIVLLMEngine에서는 락 없이 직접 감소)
            self.active_requests -= 1
            # 컨텍스트 정리
            logger.clear_context()

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
            logger.warning(f"Request ID not found in inference_stats", context={
                "request": {"id": request_id, "status": "unknown"}
            })
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
            "Request statistics",
            context={
                "stats": {
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "inference_time": stats.inference_time,
                    "tokens_per_second": stats.tokens_per_second
                }
            }
        )

        return stats

    async def get_stats(self) -> Dict[str, Any]:
        """
        엔진 성능 통계 조회

        Returns:
            성능 통계 정보
        """
        # 시작 시간 측정을 위한 타이밍 컨텍스트
        with TimingContext(logger, "Stats collection") as timing:
            try:
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

                # 메트릭 수집기에서 추가 정보 가져오기 - 모니터링이 활성화된 경우에만
                metrics_info = {}
                if config.monitoring.enabled:
                    try:
                        metrics_collector = get_metrics_collector()
                        metrics_info = metrics_collector.get_metrics()
                        logger.debug("Retrieved metrics from collector", context={
                            "metrics_collection": {"duration": timing.duration}
                        })
                    except Exception as e:
                        logger.warning(
                            "Failed to get metrics from collector",
                            context={"error": {"type": type(e).__name__, "message": str(e)}}
                        )

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

                # 프로파일링 정보 추가 - 모니터링이 활성화된 경우에만
                if config.monitoring.enabled:
                    try:
                        profiler = get_profiler()
                        memory_report = profiler.get_memory_report()
                        function_stats = profiler.get_function_stats(top_n=5)  # 상위 5개 함수만

                        memory_data = await memory_report
                        function_data = await function_stats

                        # 여기서 await 결과의 타입을 확인하고 적절히 처리
                        # 구조화된 로깅을 통해 디버그 정보 추가
                        logger.debug("Retrieved profiling data", context={
                            "profiling": {
                                "memory_report_size": len(str(memory_data)),
                                "function_stats_count": len(function_data.get("top_functions", []))
                            }
                        })

                        stats["profiling"] = {
                            "memory": {
                                "process_ram_gb": memory_data.get("current", {}).get("process_ram_used_gb", 0),
                                "system_ram_percent": memory_data.get("current", {}).get("system_ram_percent", 0)
                            },
                            "top_functions": [
                                {
                                    "name": f"{func.get('module_name', '')}.{func.get('function_name', '')}",
                                    "avg_time": func.get("avg_time", 0),
                                    "call_count": func.get("call_count", 0)
                                } for func in function_data.get("top_functions", [])[:5]
                            ]
                        }
                    except Exception as e:
                        logger.warning(
                            "Failed to get profiling stats",
                            context={
                                "error": {
                                    "type": type(e).__name__,
                                    "message": str(e),
                                    "location": "get_stats.profiling"
                                }
                            }
                        )

                return stats

            except Exception as e:
                # 전체 메서드에 걸친 예외 처리 개선
                logger.error(
                    "Error getting engine stats",
                    context={
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "location": "get_stats"
                        }
                    },
                    exc_info=True
                )
                # 최소한의 정보라도 반환
                return {
                    "model": self.model_name,
                    "error": f"Error getting stats: {str(e)}",
                    "uptime": time.time() - self.start_time,
                    "active_requests": self.active_requests
                }


# 테스트용 코드
async def main():
    """메인 함수 (테스트용)"""
    # 구조화된 로거 사용
    logger.info("Starting test main function", context={"environment": "test"})

    try:
        # 엔진 인스턴스 생성
        logger.info("Creating engine instance", context={"test": {"phase": "engine_creation"}})
        engine = MAIVLLMEngine()

        # 샘플 요청 생성
        test_config = RequestConfig(
            prompt="Hello, how are you?",
            max_tokens=100
        )

        # 추론 실행
        logger.info("Starting test inference", context={"test": {"phase": "inference"}})
        result = await engine.generate(test_config)
        print(json.dumps(result, indent=2))

        # 성능 통계 출력
        logger.info("Getting engine stats", context={"test": {"phase": "stats"}})
        stats = await engine.get_stats()
        print(json.dumps(stats, indent=2))

        logger.info("Test completed successfully", context={"test": {"status": "success"}})

    except Exception as e:
        logger.error(
            "Test failed",
            context={
                "test": {"status": "failed"},
                "error": {"type": type(e).__name__, "message": str(e)}
            },
            exc_info=True
        )
    finally:
        # 정리 작업
        logger.info("Test cleanup completed", context={"test": {"phase": "cleanup"}})


if __name__ == "__main__":
    import asyncio
    import json

    # 테스트 시작 시 로깅 레벨 설정 - 개발 환경에서는 DEBUG로 설정
    logger.info("Test mode started", context={"mode": "development", "component": "engine"})
    asyncio.run(main())
