"""
mai-vllm-serving의 핵심 서버 구현
대용량 언어 모델(LLM)을 효율적으로 서빙하기 위한 고성능 FastAPI 애플리케이션
"""

import datetime
import json
import logging
import os
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict, Any, TypedDict

import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from mai_vllm_serving.distributed import DistributedVLLMEngine, DistributedConfig
# 엔진 모듈 임포트
from mai_vllm_serving.engine import MAIVLLMEngine, RequestConfig
from mai_vllm_serving.monitoring.metrics import (
    get_metrics_collector,
    track_request,
    init_monitoring,
    get_current_metrics,
    get_request_metrics_by_id
)
from mai_vllm_serving.monitoring.profiler import init_profiling, profile, profile_block, get_profiler
# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config
from mai_vllm_serving.utils.engine_manager import EngineManager
# 로깅 유틸리티 임포트
from mai_vllm_serving.utils.logging_utils import (
    setup_logging,
    get_logger,
    get_request_logger,
    TimingContext,
    with_request_context
)

# 공유 메모리 누수 경고 무시
warnings.filterwarnings("ignore",
                        message="UserWarning: resource_tracker: "
                                "There appear to be 1 leaked shared_memory objects to clean up at shutdown")
# NCCL 프로세스 그룹 경고 필터링
warnings.filterwarnings("ignore", message=".*process group has NOT been destroyed.*")

os.environ["OMP_NUM_THREADS"] = "4"

# 설정 객체 가져오기
config = get_config()

# 로깅 성능 모드 설정 (환경 설정에 따라)
is_log_performance_mode = config.logging.log_performance_mode
if is_log_performance_mode:
    print("Enabling logging performance mode for environment")

# 로깅 초기화
root_logger = setup_logging(
    service_name="mai-vllm-serving",
    log_level=config.logging.level,
    use_json=config.logging.json,
    log_file=config.logging.file,
    include_caller_info=True,
    performance_mode=config.logging.log_performance_mode,
    async_logging=True,
    sampling_rate=config.logging.log_sampling_rate
)

# 구조화된 로거 가져오기
logger = get_logger("server", production_mode=is_log_performance_mode)

# API 요청/응답 로거 생성
request_logger = get_request_logger()


# API 요청 모델
class GenerationRequest(BaseModel):
    """LLM 생성 요청을 위한 모델"""
    prompt: str
    request_id: Optional[str] = Field(default=None, description="요청 ID (제공하지 않을 경우 자동 생성)")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, description="생성할 최대 토큰 수")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="샘플링 온도, 높을수록 더 다양한 출력")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="누적 확률 임계값")
    top_k: Optional[int] = Field(default=None, ge=0, description="샘플링할 최상위 토큰 수")
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="빈도 페널티")
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="존재 페널티")
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0, description="반복 페널티")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="생성 중단 시퀀스")
    stream: bool = Field(default=False, description="스트리밍 모드 사용 여부")

    @field_validator('prompt')
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('프롬프트는 비어있을 수 없습니다')
        return v.strip()

    @field_validator('request_id')
    def request_id_validate(cls, v):
        if v is not None and not v.strip():
            raise ValueError('request_id가 제공된 경우 비어있을 수 없습니다')
        return v.strip() if v is not None else v

    @model_validator(mode='after')
    def set_default_from_config(self):
        # 설정 객체 가져오기
        cfg = get_config()

        # 값이 None인 경우 환경설정에서 가져오기
        if self.max_tokens is None:
            self.max_tokens = getattr(cfg.inference, 'max_tokens', 4096)

        if self.temperature is None:
            self.temperature = getattr(cfg.inference, 'temperature', 0.1)

        if self.top_p is None:
            self.top_p = getattr(cfg.inference, 'top_p', 0.9)

        if self.top_k is None:
            self.top_k = getattr(cfg.inference, 'top_k', 50)

        if self.frequency_penalty is None:
            self.frequency_penalty = getattr(cfg.inference, 'frequency_penalty', 0.0)

        if self.presence_penalty is None:
            self.presence_penalty = getattr(cfg.inference, 'presence_penalty', 0.2)

        if self.repetition_penalty is None:
            self.repetition_penalty = getattr(cfg.inference, 'repetition_penalty', 1.1)

        if self.stop is None:
            self.stop = getattr(cfg.inference, 'stop', None)

        return self


class CudaInfo(TypedDict, total=False):
    available: bool
    device_count: int
    current_device: int
    current_device_name: str


class SystemInfo(TypedDict, total=False):
    cuda: CudaInfo
    memory: Dict[str, Any]


# 요청 추적 의존성
async def get_request_tracking(request: Request) -> Dict[str, Any]:
    """
    요청 추적 정보를 생성하는 의존성 함수

    Args:
        request: FastAPI 요청 객체

    Returns:
        요청 추적 정보
    """
    # 요청 ID 생성
    request_id = str(uuid.uuid4())

    # 클라이언트 정보 수집
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    # 요청 시작 시간 기록
    request_start_time = time.time()

    return {
        "request_id": request_id,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "start_time": request_start_time
    }


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """
    애플리케이션 라이프스팬 이벤트 핸들러

    Args:
        app_: FastAPI 애플리케이션
    """
    try:
        # 로깅 필터 적용 (성능 최적화)
        logging_filter = ConditionalFilter()
        app_root_logger = logging.getLogger()
        app_root_logger.addFilter(logging_filter)

        # vLLM 관련 로거의 레벨을 WARNING으로 설정
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("vllm.engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.worker").setLevel(logging.WARNING)
        logging.getLogger("vllm.model_executor").setLevel(logging.WARNING)
        logging.getLogger("vllm.executor").setLevel(logging.WARNING)
        logging.getLogger("vllm.multiproc_utils").setLevel(logging.WARNING)
        logging.getLogger("vllm.model_runner").setLevel(logging.WARNING)
        logging.getLogger("vllm.custom_all_reduce").setLevel(logging.WARNING)
        logging.getLogger("vllm.custom_all_reduce_utils").setLevel(logging.WARNING)
        # PyTorch 관련 로거
        logging.getLogger("torch").setLevel(logging.WARNING)
        # CUDA 관련 로거
        logging.getLogger("torch.cuda").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)

        logger.info("Starting mai-vllm-serving",
                    context={
                        "startup": {
                            "model": config.model.name,
                            "version": "0.1.0",
                            "environment": os.environ.get("ENV", "production"),
                            "log_level": config.logging.level
                        }
                    })

        # 분산 처리 설정 확인
        if config.distributed.world_size > 1:
            logger.info(f"Initializing distributed mode with world_size={config.distributed.world_size}")

            # 분산 설정 생성
            dist_config = DistributedConfig(
                world_size=config.distributed.world_size,
                backend=config.distributed.backend,
                master_addr=config.distributed.master_addr,
                master_port=config.distributed.master_port,
                timeout=config.distributed.timeout
            )

            # 현재 프로세스의 rank와 local_rank는 환경 변수에서 가져옴
            if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
                dist_config.rank = int(os.environ["RANK"])
                dist_config.local_rank = int(os.environ["LOCAL_RANK"])

            logger.debug(f"Distributed config: rank={dist_config.rank}, local_rank={dist_config.local_rank}",
                         context={"distributed": dist_config.to_dict()})

            # 분산 모드에서는 양자화 설정 확인
            quant_enabled = config.quantization.enabled
            if quant_enabled:
                logger.debug(
                    f"Using quantization: method={config.quantization.method}, bits={config.quantization.bits}",
                    context={"quantization": {
                        "method": config.quantization.method,
                        "bits": config.quantization.bits
                    }})

            # 분산 엔진 초기화
            with TimingContext(logger, "Distributed engine initialization") as timing:
                llm_engine = DistributedVLLMEngine(
                    model_name=config.model.name,
                    dist_config=dist_config,
                    tensor_parallel_size=config.engine.tensor_parallel_size,
                    pipeline_parallel_size=config.engine.pipeline_parallel_size,
                    gpu_memory_utilization=config.engine.gpu_memory_utilization,
                    max_num_seqs=config.engine.max_num_seqs,
                    max_num_batched_tokens=config.engine.max_num_batched_tokens,
                    quantization=config.quantization.method if quant_enabled else None,
                    swap_space=config.engine.swap_space,
                    trust_remote_code=config.model.trust_remote_code,
                    dtype=config.engine.dtype,
                    enforce_eager=config.engine.enforce_eager
                )
                EngineManager.set_instance(llm_engine)
                logger.info(f"Distributed engine initialized in {timing.duration:.2f} seconds",
                            context={"engine_init": {"duration": timing.duration, "mode": "distributed"}})
        else:
            logger.info("Initializing standard mode (non-distributed)")

            # 양자화 설정 확인
            quant_enabled = config.quantization.enabled
            if quant_enabled:
                logger.debug(
                    f"Using quantization: method={config.quantization.method}, bits={config.quantization.bits}",
                    context={"quantization": {
                        "method": config.quantization.method,
                        "bits": config.quantization.bits
                    }})

            # 표준 엔진 초기화
            with TimingContext(logger, "Engine initialization") as timing:
                llm_engine = MAIVLLMEngine(
                    model_name=config.model.name,
                    tensor_parallel_size=config.engine.tensor_parallel_size,
                    gpu_memory_utilization=config.engine.gpu_memory_utilization,
                    max_num_seqs=config.engine.max_num_seqs,
                    max_num_batched_tokens=config.engine.max_num_batched_tokens,
                    quantization=config.quantization.method if quant_enabled else None,
                    swap_space=config.engine.swap_space,
                    trust_remote_code=config.model.trust_remote_code,
                    dtype=config.engine.dtype,
                    disable_log_stats=config.engine.disable_log_stats,
                    enforce_eager=config.engine.enforce_eager
                )
                EngineManager.set_instance(llm_engine)
                logger.info(f"Engine initialized in {timing.duration:.2f} seconds",
                            context={"engine_init": {"duration": timing.duration, "mode": "standard"}})

        # 모니터링 시스템 초기화 (enabled 체크는 함수 내부에서 수행)
        monitor_enabled = config.monitoring.enabled
        if monitor_enabled:
            init_monitoring()
            # 프로파일링 시스템 초기화
            init_profiling()
            logger.info("Monitoring and profiling systems initialized",
                        context={
                            "monitoring": {"enabled": True, "profile_interval": config.monitoring.profile_interval}})
        else:
            logger.info("Monitoring and profiling systems are disabled")

        # 서버 시작 시간 계산
        startup_duration = time.time() - startup_time
        logger.info(f"Server startup completed in {startup_duration:.2f} seconds",
                    context={"startup": {"duration": startup_duration}})

    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}",
                     context={"error": {"type": type(e).__name__, "message": str(e)}},
                     exc_info=True)
        # 초기화 실패해도 서버는 시작되지만, API 호출 시 오류 반환

    yield  # 이 부분에서 애플리케이션 실행

    # 종료 이벤트 (이전의 shutdown_event 내용)
    shutdown_start_time = time.time()
    logger.info("Starting mai-vllm-serving shutdown process")

    engine = EngineManager.get_instance()
    if engine is not None:
        logger.info("Shutting down mai-vllm-serving engine")

        # 모니터링 시스템 정리
        if config.monitoring.enabled:
            try:
                metrics_collector = get_metrics_collector()
                metrics_collector.stop_collection()
                logger.info("Metrics collection stopped")

                # 프로파일러 정리 추가
                try:
                    profiler = get_profiler()
                    profiler.stop()
                    logger.info("Profiler stopped")
                except Exception as e:
                    logger.warning(f"Failed to stop profiler: {str(e)}",
                                   context={"error": {"component": "profiler", "message": str(e)}})
            except Exception as e:
                logger.warning(f"Failed to stop metrics collection: {str(e)}",
                               context={"error": {"component": "metrics", "message": str(e)}})

        # 분산 처리 종료
        if isinstance(engine, DistributedVLLMEngine):
            engine.shutdown()
            logger.info("Distributed engine shutdown completed")

        # CUDA 캐시 정리 (분산 여부와 상관없이)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {str(e)}",
                               context={"error": {"component": "cuda", "message": str(e)}})

        # 종료 시간 계산
        shutdown_duration = time.time() - shutdown_start_time
        logger.info(f"mai-vllm-serving shutdown completed in {shutdown_duration:.2f} seconds",
                    context={"shutdown": {"duration": shutdown_duration}})


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="mai-vllm-serving",
    description="고성능 LLM 서빙 API",
    version="0.1.0",
    default_response_class=JSONResponse,  # JSON 응답에 한글 처리를 위한 기본 응답 클래스 지정
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JSON 응답에서 한글 처리를 위한 미들웨어 추가 (선택적)
@app.middleware("http")
async def add_utf8_header(request: Request, call_next):
    """UTF-8 인코딩 헤더 추가 미들웨어"""
    # 성능 모드 감지
    performance_mode = config.logging.log_performance_mode

    # 요청 로깅 확장
    req_method = request.method
    req_url = str(request.url)
    client_info = f"{request.client.host}:{request.client.port}" if request.client else "unknown"

    # 성능 모드 여부에 따라 로깅 수준 조정
    if performance_mode:
        # 성능 모드에서는 INFO 이상의 엔드포인트만 로깅
        should_log = req_url.endswith(("/generate", "/health", "/metrics"))
        if should_log:
            logger.debug(f"HTTP Request: {req_method} {req_url}",
                         context={
                             "http": {
                                 "method": req_method,
                                 "url": req_url,
                                 "client": client_info
                             }
                         })
    else:
        # DEBUG 레벨에서 모든 요청 로깅
        logger.debug(f"HTTP Request: {req_method} {req_url}",
                     context={
                         "http": {
                             "method": req_method,
                             "url": req_url,
                             "client": client_info,
                             "user_agent": request.headers.get("user-agent", "unknown")
                         }
                     })

    # 요청 처리 및 응답 시간 측정
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # 스트리밍 응답에 대해 Content-Type 헤더 설정
    if "text/event-stream" in response.headers.get("content-type", ""):
        response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    else:
        response.headers["Content-Type"] = "application/json; charset=utf-8"

    # 응답 시간 헤더 추가 (디버깅 용도)
    response.headers["X-Process-Time"] = str(process_time)

    # 처리 완료된 요청 로깅 (200 응답은 DEBUG, 에러는 INFO 이상으로 로깅)
    if performance_mode:
        # 성능 모드: 오류 응답 또는 느린 응답(500ms 이상)만 INFO 로깅
        if response.status_code >= 400 or process_time > 0.5:
            logger.info(f"HTTP Response: {response.status_code} in {process_time:.3f}s",
                        context={
                            "http": {
                                "method": req_method,
                                "url": req_url,
                                "status_code": response.status_code,
                                "process_time": process_time,
                                "client": client_info
                            }
                        })
    else:
        # 개발 모드: 일반 응답은 DEBUG, 오류는 INFO로 로깅
        if 200 <= response.status_code < 400:
            logger.debug(f"HTTP Response: {response.status_code} in {process_time:.3f}s",
                         context={
                             "http": {
                                 "method": req_method,
                                 "url": req_url,
                                 "status_code": response.status_code,
                                 "process_time": process_time
                             }
                         })
        else:
            logger.info(f"HTTP Error Response: {response.status_code} in {process_time:.3f}s",
                        context={
                            "http": {
                                "method": req_method,
                                "url": req_url,
                                "status_code": response.status_code,
                                "process_time": process_time,
                                "client": client_info
                            }
                        })

    return response


@app.get("/metrics")
async def metrics():
    """서버 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        logger.debug("Metrics endpoint accessed but monitoring is disabled")
        return {"status": "monitoring_disabled", "message": "Monitoring is disabled in configuration"}

    try:
        logger.debug("Metrics endpoint accessed")
        metrics_data = await get_current_metrics()
        return metrics_data
    except Exception as e:
        error_msg = f"Error getting metrics: {str(e)}"
        logger.error(error_msg,
                     context={"error": {"type": type(e).__name__, "message": str(e)}},
                     exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


# 프로파일링 결과를 위한 새 엔드포인트 추가
@app.get("/metrics/profiling")
async def profiling_metrics():
    """프로파일링 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        logger.debug("Profiling metrics endpoint accessed but monitoring is disabled")
        return {"status": "monitoring_disabled", "message": "Monitoring is disabled in configuration"}

    try:
        logger.debug("Profiling metrics endpoint accessed")
        profiler = get_profiler()
        return {
            "memory": await profiler.get_memory_report(),
            "functions": await profiler.get_function_stats(),
            "system": await profiler.get_system_stats()
        }
    except Exception as e:
        error_msg = f"Error getting profiling metrics: {str(e)}"
        logger.error(error_msg,
                     context={"error": {"type": type(e).__name__, "message": str(e)}},
                     exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/metrics/request/{request_id}")
async def request_metrics(request_id: str):
    """특정 요청의 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        logger.debug(f"Request metrics endpoint accessed for {request_id} but monitoring is disabled")
        return {"status": "monitoring_disabled", "message": "Monitoring is disabled in configuration"}

    try:
        logger.debug(f"Request metrics endpoint accessed for request ID: {request_id}")
        metrics_data = await get_request_metrics_by_id(request_id)
        if metrics_data is None:
            logger.warning(f"Request metrics not found for request ID: {request_id}")
            raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
        return metrics_data
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error getting request metrics: {str(e)}"
        logger.error(error_msg,
                     context={"error": {"type": type(e).__name__, "message": str(e), "request_id": request_id}},
                     exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/generate")
@track_request
@profile
@with_request_context
async def generate(
        request: GenerationRequest,
        background_tasks: BackgroundTasks,
        client_request: Request,
        tracking: Dict[str, Any] = Depends(get_request_tracking)
):
    """
    텍스트 생성 엔드포인트

    Args:
        request: 생성 요청 파라미터
        background_tasks: 백그라운드 작업 처리
        client_request: FastAPI 요청 객체
        tracking: 요청 추적 정보

    Returns:
        생성된 텍스트와 관련 메타데이터
    """
    # 성능 모드 감지
    performance_mode = config.logging.log_performance_mode

    # 요청 ID 생성 및 로깅
    request_id = request.request_id if request.request_id else tracking["request_id"]
    client_ip = client_request.client.host if client_request.client else "unknown"
    request_data = request.model_dump()

    # 실제 사용될 request_id를 request_data에 업데이트
    request_data['request_id'] = request_id

    # 요청 컨텍스트 설정
    logger.set_request_id(request_id)

    # 로깅 최적화: 성능 모드에서는 간결한 로깅
    if performance_mode:
        logger.info(f"Generation request received: {request_id}",
                    context={
                        "request": {
                            "id": request_id,
                            "prompt_length": len(request.prompt),
                            "stream": request.stream
                        }
                    })
    else:
        # 상세한 요청 로깅
        logger.info(f"Generation request received: {request_id}",
                    context={
                        "request": {
                            "id": request_id,
                            "prompt_length": len(request.prompt),
                            "max_tokens": request.max_tokens,
                            "temperature": request.temperature,
                            "client_ip": client_ip,
                            "user_agent": client_request.headers.get("user-agent", "unknown"),
                            "stream": request.stream
                        }
                    })

    # 요청 로깅 - 요청 데이터 마스킹이 정책적으로 필요한 경우 이 함수가 처리
    request_logger.log_request(
        request_data=request_data,
        request_id=request_id,
        client_ip=client_ip
    )

    # 타이밍 컨텍스트 시작
    timing = TimingContext(logger, f"Request {request_id} processing",
                           log_threshold=0.1 if performance_mode else None)

    try:
        with timing:
            # 엔진용 RequestConfig 생성
            engine_request = RequestConfig(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop,
                stream=request.stream,
                request_id=request_id,
                seed=config.inference.seed
            )

            # 스트리밍 모드인 경우
            if request.stream:
                # 백그라운드에서 요청 종료 로깅
                background_tasks.add_task(
                    lambda msg, req_id: logger.info(msg, context={"request_id": req_id}),
                    f"Streaming request started: {request_id}",
                    request_id
                )

                # 스트리밍 응답 생성
                with profile_block(f"stream_response_{request_id}"):  # 프로파일링 블록 추가
                    logger.debug(f"Starting streaming response",
                                 context={"streaming": {"request_id": request_id}})
                    return StreamingResponse(
                        _stream_response_generator(engine_request, timing),
                        media_type="text/event-stream"
                    )

            # 일반 모드인 경우 - 엔진에 요청 전달
            with profile_block(f"generate_request_{request_id}"):  # 프로파일링 블록 추가
                llm_engine = EngineManager.get_instance()
                logger.debug(f"Starting generation request",
                             context={"generation": {"request_id": request_id}})
                result = await llm_engine.generate(engine_request)

            # 응답 로깅
            response_log_data = {
                "request_id": request_id,
                "generated_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": result.get("usage", {}).get("total_tokens", 0),
                "inference_time": result.get("performance", {}).get("inference_time", "0"),
                "tokens_per_second": result.get("performance", {}).get("tokens_per_second", "0"),
                "finish_reason": result.get("finish_reason", "unknown")
            }

            logger.info(f"Generation request completed",
                        context={"response": response_log_data})

            request_logger.log_response(
                response_data=result,
                status_code=200,
                processing_time=timing.duration,
                request_id=request_id
            )

            return JSONResponse(content=result)

    except Exception as e:
        error_context = {
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "request_id": request_id
            }
        }

        logger.error("Error processing generation request",
                     context=error_context,
                     exc_info=True)

        # 오류 로깅
        request_logger.log_error(
            error=e,
            request_data=request_data,
            status_code=500,
            request_id=request_id
        )

        # 요청 ID가 설정된 HTTP 예외 반환
        detail = {"error": str(e), "request_id": request_id, "timestamp": datetime.datetime.now().isoformat()}
        raise HTTPException(status_code=500, detail=detail)


async def _stream_response_generator(engine_request: RequestConfig, timing: TimingContext):
    """
    스트리밍 응답 생성기

    Args:
        engine_request: 엔진 요청 구성
        timing: 타이밍 컨텍스트

    Yields:
        SSE 형식의 생성 결과
    """
    request_id = engine_request.request_id
    logger.debug(f"Stream generator started",
                 context={"streaming": {"request_id": request_id}})

    token_count = 0
    stream_start_time = time.time()

    try:
        # 엔진에서 스트리밍 응답 반환 - 제너레이터이므로 그대로 사용 가능
        llm_engine = EngineManager.get_instance()
        async for chunk in await llm_engine.generate(engine_request):
            # 토큰 카운트 업데이트 (new_text가 있는 경우)
            if "new_text" in chunk:
                token_count += 1  # 정확한 토큰 수를 모르기 때문에 청크 수로 근사

            # UTF-8 인코딩을 보장하는 JSON 직렬화
            response_json = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {response_json}\n\n"

            # 완료 시 로깅
            if chunk.get("finished", False):
                stream_duration = time.time() - stream_start_time
                tokens_per_second = token_count / stream_duration if stream_duration > 0 else 0

                logger.info(f"Streaming request completed: {request_id}",
                            context={
                                "streaming": {
                                    "request_id": request_id,
                                    "duration": stream_duration,
                                    "tokens": token_count,
                                    "tokens_per_second": tokens_per_second,
                                    "finish_reason": chunk.get("finish_reason", "unknown")
                                }
                            })

    except Exception as e:
        error_context = {
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "request_id": request_id
            }
        }

        logger.error(f"Error in stream generation",
                     context=error_context,
                     exc_info=True)

        error_json = json.dumps({
            "id": engine_request.request_id,
            "error": str(e),
            "finished": True,
            "timestamp": datetime.datetime.now().isoformat()
        }, ensure_ascii=False)

        yield f"data: {error_json}\n\n"

    # 스트림 종료
    logger.debug(f"Stream generator completed",
                 context={"streaming": {"request_id": request_id}})
    yield f"data: [DONE]\n\n"


@app.get("/health")
async def health():
    """
    서버 상태 확인 엔드포인트

    Returns:
        서버 상태 정보
    """
    llm_engine = EngineManager.get_instance()

    # 상태 검사 시작을 로깅
    logger.debug("Health check requested")

    if llm_engine is None:
        logger.warning("Health check requested but server is not initialized")
        return {"status": "initializing", "timestamp": datetime.datetime.now().isoformat()}

    try:
        # 엔진에서 상태 정보 가져오기
        engine_stats = await llm_engine.get_stats()

        # 모니터링 상태 추가
        monitoring_status = "enabled" if config.monitoring.enabled else "disabled"

        # 분산 처리 정보 포함
        is_distributed = isinstance(llm_engine, DistributedVLLMEngine)

        # 시스템 리소스 정보 추가
        system_info: SystemInfo = {}
        if torch.cuda.is_available():
            system_info["cuda"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "current_device_name": torch.cuda.get_device_name(torch.cuda.current_device())
            }
        else:
            system_info["cuda"] = {"available": False}

        # 메모리 정보
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["memory"] = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "percent_used": memory.percent
            }
        except ImportError:
            system_info["memory"] = {"available": False}

        # 상태 정보 구성
        status_info = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "model": config.model.name,
            "uptime": engine_stats.get("uptime", "unknown"),
            "monitoring": monitoring_status,
            "is_distributed": is_distributed,
            "server_config": {
                "host": config.server.host,
                "port": config.server.port,
                "workers": config.server.workers
            },
            "system_info": system_info,
            "engine_stats": engine_stats
        }

        logger.debug("Health check responding with healthy status",
                     context={"health_check": {"status": "healthy"}})
        return status_info

    except Exception as e:
        error_context = {
            "error": {
                "type": type(e).__name__,
                "message": str(e)
            }
        }

        logger.error(f"Error during health check",
                     context=error_context,
                     exc_info=True)

        return {
            "status": "degraded",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.get("/")
async def root():
    """
    루트 엔드포인트

    Returns:
        서버 정보
    """
    logger.debug("Root endpoint accessed")

    engine_status = "running" if EngineManager.get_instance() is not None else "initializing"
    server_uptime = time.time() - globals().get("startup_time", time.time())

    return {
        "service": "mai-vllm-serving",
        "version": "0.1.0",
        "status": engine_status,
        "model": config.model.name,
        "uptime_seconds": round(server_uptime, 2),
        "environment": os.environ.get("ENV", "production"),
        "documentation": "/docs",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    전역 예외 처리기

    모든 처리되지 않은 예외를 적절하게 로깅하고 응답

    Args:
        request: FastAPI 요청 객체
        exc: 발생한 예외

    Returns:
        적절한 오류 응답
    """
    # 성능 모드 감지
    performance_mode = config.logging.log_performance_mode

    # 요청 정보 수집
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"

    # 오류 컨텍스트 구성 (성능 모드에서는 경량화)
    if performance_mode:
        error_context = {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "request_id": request_id
            }
        }
    else:
        error_context = {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "client_ip": client_ip
            }
        }

    # HTTPException은 일반적인 오류이므로 INFO 레벨로 로깅
    if isinstance(exc, HTTPException):
        if exc.status_code >= 500:
            logger.error(f"HTTP Error {exc.status_code}", context=error_context)
        else:
            logger.info(f"HTTP Error {exc.status_code}", context=error_context)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    # 처리되지 않은 예외는 ERROR 레벨로 로깅
    logger.error("Unhandled exception", context=error_context, exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "request_id": request_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )


# 서버 시작 시간 저장
startup_time = time.time()


def main():
    """메인 함수"""
    # 설정에서 서버 설정 가져오기
    host = config.server.host
    port = config.server.port
    workers = config.server.workers
    log_level = config.server.log_level
    timeout = config.server.request_timeout

    # 서버 시작 정보 로깅
    logger.info("Starting mai-vllm-serving server",
                context={
                    "server": {
                        "host": host,
                        "port": port,
                        "workers": workers,
                        "timeout": timeout,
                        "log_level": log_level,
                        "model": config.model.name,
                        "environment": os.environ.get("ENV", "production")
                    }
                })

    # 분산 처리 모드인 경우 워커 수 확인
    if config.distributed.world_size > 1 and workers > 1:
        logger.warning(
            f"Distributed mode with world_size={config.distributed.world_size} is enabled, "
            f"but server workers={workers} > 1. This may cause conflicts.",
            context={
                "warning": {
                    "type": "configuration",
                    "distributed_world_size": config.distributed.world_size,
                    "server_workers": workers,
                    "recommendation": "Set workers=1 when using distributed mode"
                }
            }
        )

    # FastAPI 서버 시작
    if workers > 1:
        # 애플리케이션을 임포트 문자열로 실행
        import subprocess
        import sys

        logger.info(f"Starting server with {workers} workers")

        cmd = [
            sys.executable, "-m", "uvicorn",
            "mai_vllm_serving.server:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers),
            "--log-level", log_level.lower(),
            "--timeout-keep-alive", str(timeout)
        ]

        try:
            subprocess.run(cmd)
        except Exception as e:
            logger.critical(f"Failed to start server: {str(e)}",
                            context={"error": {"type": type(e).__name__, "message": str(e)}},
                            exc_info=True)
            raise
    else:
        # 단일 워커인 경우 직접 실행
        logger.info("Starting server with single worker")

        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                timeout_keep_alive=timeout
            )
        except Exception as e:
            logger.critical(f"Failed to start server: {str(e)}",
                            context={"error": {"type": type(e).__name__, "message": str(e)}},
                            exc_info=True)
            raise


if __name__ == "__main__":
    main()
