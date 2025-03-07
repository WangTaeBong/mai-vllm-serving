#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mai-vllm-serving의 핵심 서버 구현
대용량 언어 모델(LLM)을 효율적으로 서빙하기 위한 고성능 FastAPI 애플리케이션
"""

import json
import locale
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

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
    get_request_logger,
    TimingContext
)

# 설정 객체 가져오기
config = get_config()

# 로깅 초기화
logger = setup_logging(
    service_name="mai-vllm-serving",
    log_level=config.logging.level,
    use_json=config.logging.json,
    log_file=config.logging.file
)

# 요청/응답 로거 가져오기
request_logger = get_request_logger()


# API 요청 모델
class GenerationRequest(BaseModel):
    """LLM 생성 요청을 위한 모델"""
    prompt: str
    max_tokens: int = Field(default=config.inference.max_tokens, ge=1, le=8192, description="생성할 최대 토큰 수")
    temperature: float = Field(default=config.inference.temperature, ge=0.0, le=2.0,
                               description="샘플링 온도, 높을수록 더 다양한 출력")
    top_p: float = Field(default=config.inference.top_p, ge=0.0, le=1.0, description="누적 확률 임계값")
    top_k: int = Field(default=config.inference.top_k, ge=0, description="샘플링할 최상위 토큰 수")
    frequency_penalty: float = Field(default=config.inference.frequency_penalty, ge=0.0, le=2.0, description="빈도 페널티")
    presence_penalty: float = Field(default=config.inference.presence_penalty, ge=0.0, le=2.0, description="존재 페널티")
    repetition_penalty: float = Field(default=config.inference.repetition_penalty, ge=1.0, le=2.0, description="반복 페널티")
    no_repeat_ngram_size: int = Field(default=config.inference.no_repeat_ngram_size, ge=0,
                                      description="반복하지 않을 n-gram 크기")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="생성 중단 시퀀스")
    stream: bool = Field(default=False, description="스트리밍 모드 사용 여부")

    @field_validator('prompt')
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('프롬프트는 비어있을 수 없습니다')
        return v.strip()


# 한글 인코딩 설정
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, '.UTF-8')
    except locale.Error:
        logger.warning("Failed to set Korean locale. Falling back to default locale.")

# 기본 인코딩 확인 및 설정
if sys.stdout.encoding != 'utf-8':
    logger.warning(f"System stdout encoding is {sys.stdout.encoding}, recommended UTF-8")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """
    애플리케이션 라이프스팬 이벤트 핸들러

    Args:
        app_: FastAPI 애플리케이션
    """
    try:
        logger.info("Starting mai-vllm-serving")
        logger.info(f"Loading model: {config.model.name}")

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

            logger.info(f"Distributed config: rank={dist_config.rank}, local_rank={dist_config.local_rank}")

            # 분산 모드에서는 양자화 설정 확인
            quant_enabled = config.quantization.enabled
            if quant_enabled:
                logger.info(f"Using quantization: method={config.quantization.method}, bits={config.quantization.bits}")

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
                logger.info(f"Distributed engine initialized in {timing.duration:.2f} seconds")
        else:
            logger.info("Initializing standard mode (non-distributed)")

            # 양자화 설정 확인
            quant_enabled = config.quantization.enabled
            if quant_enabled:
                logger.info(f"Using quantization: method={config.quantization.method}, bits={config.quantization.bits}")

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
                logger.info(f"Engine initialized in {timing.duration:.2f} seconds")

        # 모니터링 시스템 초기화
        if config.monitoring.enabled:
            init_monitoring()
            # 프로파일링 시스템 초기화
            init_profiling()
            logger.info("Monitoring and profiling systems initialized")

        logger.info("mai-vllm-serving is ready to process requests")

    except Exception as e:
        logger.error(f"Failed to initialize LLM engine: {str(e)}", exc_info=True)
        # 초기화 실패해도 서버는 시작되지만, API 호출 시 오류 반환

    yield  # 이 부분에서 애플리케이션 실행

    # 종료 이벤트 (이전의 shutdown_event 내용)
    engine = EngineManager.get_instance()
    if engine is not None:
        logger.info("Shutting down mai-vllm-serving")

        # 모니터링 시스템 정리
        if config.monitoring.enabled:
            try:
                metrics_collector = get_metrics_collector()
                metrics_collector.stop_collection()
                logger.info("Metrics collection stopped")
            except Exception as e:
                logger.warning(f"Failed to stop metrics collection: {str(e)}")

        # 분산 처리 종료
        if isinstance(engine, DistributedVLLMEngine):
            engine.shutdown()

        logger.info("mai-vllm-serving shutdown completed")


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
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.get("/metrics")
async def metrics():
    """서버 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        return {"status": "monitoring_disabled"}

    try:
        metrics_data = await get_current_metrics()
        return metrics_data
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


# 프로파일링 결과를 위한 새 엔드포인트 추가
@app.get("/metrics/profiling")
async def profiling_metrics():
    """프로파일링 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        return {"status": "monitoring_disabled"}

    try:
        profiler = get_profiler()
        return {
            "memory": await profiler.get_memory_report(),
            "functions": await profiler.get_function_stats(),
            "system": await profiler.get_system_stats()
        }
    except Exception as e:
        logger.error(f"Error getting profiling metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting profiling metrics: {str(e)}")


@app.get("/metrics/request/{request_id}")
async def request_metrics(request_id: str):
    """특정 요청의 메트릭 정보 반환"""
    if not config.monitoring.enabled:
        return {"status": "monitoring_disabled"}

    try:
        metrics_data = await get_request_metrics_by_id(request_id)
        if metrics_data is None:
            raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
        return metrics_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting request metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting request metrics: {str(e)}")


@app.post("/generate")
@track_request
@profile
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks, client_request: Request):
    """
    텍스트 생성 엔드포인트

    Args:
        request: 생성 요청 파라미터
        background_tasks: 백그라운드 작업 처리
        client_request: FastAPI 요청 객체

    Returns:
        생성된 텍스트와 관련 메타데이터
    """
    # global llm_engine  # 명시적으로 글로벌 변수 참조

    # 요청 ID 생성 및 로깅
    request_id = str(uuid.uuid4())
    client_ip = client_request.client.host if client_request.client else "unknown"
    request_data = request.dict()

    request_logger.log_request(
        request_data=request_data,
        request_id=request_id,
        client_ip=client_ip
    )

    # 타이밍 컨텍스트 시작
    timing = TimingContext(logger, f"Request {request_id} processing")

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
                # no_repeat_ngram_size=request.no_repeat_ngram_size,
                stop=request.stop,
                stream=request.stream,
                request_id=request_id,
                seed=config.inference.seed
            )

            # 스트리밍 모드인 경우
            if request.stream:
                # 백그라운드에서 요청 종료 로깅
                background_tasks.add_task(
                    lambda msg: logger.info(msg),
                    f"Streaming request {request_id} started"
                )

                # 스트리밍 응답 생성
                with profile_block(f"stream_response_{request_id}"):  # 프로파일링 블록 추가
                    return StreamingResponse(
                        _stream_response_generator(engine_request, timing),
                        media_type="text/event-stream"
                    )

            # 일반 모드인 경우 - 엔진에 요청 전달
            with profile_block(f"generate_request_{request_id}"):  # 프로파일링 블록 추가
                llm_engine = EngineManager.get_instance()
                result = await llm_engine.generate(engine_request)

            # 응답 로깅
            request_logger.log_response(
                response_data=result,
                status_code=200,
                processing_time=timing.duration,
                request_id=request_id
            )

            return JSONResponse(content=result)

    except Exception as e:
        # 오류 로깅
        request_logger.log_error(
            error=e,
            request_data=request_data,
            status_code=500,
            request_id=request_id
        )
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_response_generator(engine_request: RequestConfig, timing: TimingContext):
    """
    스트리밍 응답 생성기

    Args:
        engine_request: 엔진 요청 구성
        timing: 타이밍 컨텍스트

    Yields:
        SSE 형식의 생성 결과
    """
    try:
        # 엔진에서 스트리밍 응답 반환 - 제너레이터이므로 그대로 사용 가능
        llm_engine = EngineManager.get_instance()
        async for chunk in await llm_engine.generate(engine_request):
            response_json = json.dumps(chunk)
            yield f"data: {response_json}\n\n"

            # 완료 시 로깅
            if chunk.get("finished", False):
                logger.info(
                    f"Streaming request {engine_request.request_id} completed in {timing.duration:.3f}s"
                )

    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
        error_json = json.dumps({
            "id": engine_request.request_id,
            "error": str(e),
            "finished": True
        })
        yield f"data: {error_json}\n\n"

    # 스트림 종료
    yield f"data: [DONE]\n\n"


@app.get("/health")
async def health():
    """
    서버 상태 확인 엔드포인트

    Returns:
        서버 상태 정보
    """
    llm_engine = EngineManager.get_instance()
    if llm_engine is None:
        logger.warning("Health check requested but server is not initialized")
        return {"status": "initializing"}

    try:
        # 엔진에서 상태 정보 가져오기
        engine_stats = await llm_engine.get_stats()

        # 모니터링 상태 추가
        monitoring_status = "enabled" if config.monitoring.enabled else "disabled"

        # 분산 처리 정보 포함
        is_distributed = isinstance(llm_engine, DistributedVLLMEngine)

        # 상태 정보 구성
        status_info = {
            "status": "healthy",
            "model": config.model.name,
            "uptime": engine_stats.get("uptime", "unknown"),
            "monitoring": monitoring_status,
            "is_distributed": is_distributed,
            "server_config": {
                "host": config.server.host,
                "port": config.server.port,
                "workers": config.server.workers
            },
            "engine_stats": engine_stats
        }

        logger.debug(f"Health check responded with: {json.dumps(status_info)}")
        return status_info

    except Exception as e:
        logger.error(f"Error during health check: {str(e)}", exc_info=True)
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/")
async def root():
    """
    루트 엔드포인트

    Returns:
        서버 정보
    """
    return {
        "service": "mai-vllm-serving",
        "version": "0.1.0",
        "status": "running" if EngineManager.get_instance() is not None else "initializing",
        "model": config.model.name,
        "documentation": "/docs"
    }


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

    # 로깅 설정 최종 업데이트
    logger.info(f"Starting mai-vllm-serving on {host}:{port} with {workers} worker(s)")
    logger.info(f"Model: {config.model.name}")

    # 분산 처리 모드인 경우 워커 수 확인
    if config.distributed.world_size > 1 and workers > 1:
        logger.warning(
            f"Distributed mode with world_size={config.distributed.world_size} is enabled, "
            f"but server workers={workers} > 1. This may cause conflicts. "
            "Consider setting workers=1 when using distributed mode."
        )

    # FastAPI 서버 시작
    if workers > 1:
        # 애플리케이션을 임포트 문자열로 실행
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "uvicorn",
            "mai_vllm_serving.server:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers),
            "--log-level", log_level.lower(),
            "--timeout-keep-alive", str(timeout)
        ]
        subprocess.run(cmd)
    else:
        # 단일 워커인 경우 직접 실행
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            timeout_keep_alive=timeout
        )


if __name__ == "__main__":
    main()
