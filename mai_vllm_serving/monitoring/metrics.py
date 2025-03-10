"""
mai-vllm-serving의 모니터링 지표 모듈
성능, 리소스 사용량, 요청 처리 등에 관한 지표를 수집하고 관리
"""

import functools
import statistics
import threading
import time
import uuid
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Deque

import psutil
import torch

from mai_vllm_serving.utils.logging_utils import with_logging_context

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    from prometheus_client import Counter as PromCounter
    from prometheus_client import Gauge, Histogram, Summary
    from prometheus_client import start_http_server as start_prometheus_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config
# 구조화 로깅 임포트
from mai_vllm_serving.utils.logging_utils import (
    get_logger,
    TimingContext
)

# 설정 객체 가져오기
config = get_config()

# 성능 모드 여부 확인
performance_mode = config.logging.log_performance_mode

# 구조화된 로거 가져오기
logger = get_logger("metrics", production_mode=performance_mode)


@dataclass
class RequestMetrics:
    """개별 요청에 대한 메트릭 정보"""
    request_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    queue_time: float = 0.0  # 큐에서 대기한 시간
    processing_time: float = 0.0  # 실제 처리 시간 (큐 대기 시간 제외)
    prompt_processing_time: float = 0.0  # 프롬프트 처리 시간
    completion_time: float = 0.0  # 토큰 생성 시간
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    client_ip: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """총 처리 시간"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """초당 생성 토큰 수"""
        if self.completion_time <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.completion_time

    def to_dict(self) -> Dict[str, Any]:
        """메트릭을 딕셔너리로 변환"""
        result = asdict(self)
        result["total_time"] = self.total_time
        result["tokens_per_second"] = self.tokens_per_second
        return result

    def complete(self, completion_tokens: int, prompt_tokens: Optional[int] = None) -> None:
        """요청 완료 처리"""
        self.end_time = time.time()
        self.completion_tokens = completion_tokens
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.completion_time = self.total_time - self.prompt_processing_time - self.queue_time
        self.status = "completed"

        # 로깅을 위한 정보 준비
        tokens_per_second = self.tokens_per_second
        total_time = self.total_time

        return {
            "completion_tokens": completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "total_time": total_time,
            "tokens_per_second": tokens_per_second
        }

    def fail(self, error: str) -> None:
        """요청 실패 처리"""
        self.end_time = time.time()
        self.error = error
        self.status = "failed"

        return {
            "error": error,
            "total_time": self.total_time
        }


@dataclass
class SystemMetrics:
    """시스템 자원 사용에 관한 메트릭 정보"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_percent: float = 0.0

    @staticmethod
    def collect() -> 'SystemMetrics':
        """현재 시스템 상태 수집"""
        metrics = SystemMetrics()

        # CPU 사용량
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

        # RAM 사용량
        mem = psutil.virtual_memory()
        metrics.ram_used_gb = mem.used / (1024 ** 3)
        metrics.ram_total_gb = mem.total / (1024 ** 3)
        metrics.ram_percent = mem.percent

        # 디스크 사용량
        disk = psutil.disk_usage('/')
        metrics.disk_used_gb = disk.used / (1024 ** 3)
        metrics.disk_total_gb = disk.total / (1024 ** 3)
        metrics.disk_percent = disk.percent

        # GPU 정보 (사용 가능한 경우)
        if HAS_GPUTIL and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_metrics = {
                        "id": i,
                        "name": gpu.name,
                        "memory_used_percent": gpu.memoryUtil * 100,
                        "memory_used_gb": gpu.memoryUsed / 1024,
                        "memory_total_gb": gpu.memoryTotal / 1024,
                        "temperature": gpu.temperature,
                        "gpu_utilization": gpu.load * 100
                    }
                    metrics.gpu_metrics.append(gpu_metrics)
            except Exception as e:
                logger.warning(
                    "GPUtil 메트릭 수집 실패",
                    context={
                        "error": str(e),
                        "component": "SystemMetrics.collect"
                    }
                )

        # PyTorch CUDA 정보
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_metrics = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024 ** 3),
                    }
                    # 이미 GPUtil에서 추가된 GPU와 합치기
                    if len(metrics.gpu_metrics) > i:
                        metrics.gpu_metrics[i].update(gpu_metrics)
                    else:
                        metrics.gpu_metrics.append(gpu_metrics)
                except Exception as e:
                    logger.warning(
                        f"PyTorch CUDA 메트릭 수집 실패 (GPU {i})",
                        context={
                            "error": str(e),
                            "gpu_id": i,
                            "component": "SystemMetrics.collect"
                        }
                    )

        return metrics


class MetricsCollector:
    """
    메트릭 수집 및 관리 클래스
    """

    def __init__(self,
                 max_history: int = 1000,
                 enable_prometheus: bool = True,
                 prometheus_port: int = 8001,
                 log_interval: int = 60):
        """
        메트릭 수집기 초기화

        Args:
            max_history: 이력 보관 최대 항목 수
            enable_prometheus: Prometheus 익스포트 활성화 여부
            prometheus_port: Prometheus 서버 포트
            log_interval: 주기적 로깅 간격 (초)
        """
        self.max_history = max_history
        self.request_metrics: Dict[str, RequestMetrics] = {}

        # 성능 모드에서는 이력 크기 감소
        history_size = max_history // 2 if performance_mode else max_history
        self.request_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self.system_metrics_history: Deque[SystemMetrics] = deque(maxlen=history_size)
        self.start_time = time.time()

        # 통계 정보
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        # 성능 통계 - 성능 모드에서는 더 작은 이력 유지
        sample_size = max_history // 4 if performance_mode else max_history
        self.request_times: Deque[float] = deque(maxlen=sample_size)
        self.tokens_per_second: Deque[float] = deque(maxlen=sample_size)

        # 오류 통계
        self.error_counter = Counter()

        # 스레드 잠금
        self.lock = threading.RLock()

        # 프로메테우스 지표 (사용 가능한 경우)
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.prometheus_port = prometheus_port
        self.prometheus_metrics = {}

        # 주기적 작업 간격 조정 (성능 모드에서는 더 긴 간격)
        self.log_interval = log_interval * 2 if performance_mode else log_interval
        self._stop_event = threading.Event()
        self._collection_thread = None

        # 메트릭 초기화 상태 추적을 위한 변수 추가
        self._metrics_initialized = False

        # 초기화 로깅
        if performance_mode:
            logger.info("메트릭 수집기 초기화 (성능 모드)")
        else:
            logger.info(
                "메트릭 수집기 초기화",
                context={
                    "max_history": max_history,
                    "enable_prometheus": self.enable_prometheus,
                    "prometheus_port": prometheus_port,
                    "log_interval": log_interval,
                }
            )

        # 프로메테우스 설정
        if self.enable_prometheus:
            with TimingContext(logger, "Prometheus 메트릭 초기화") as timing:
                self._setup_prometheus_metrics()
                start_prometheus_server(self.prometheus_port)
                logger.info(
                    "Prometheus 메트릭 서버 시작됨",
                    context={
                        "port": self.prometheus_port,
                        "init_time": timing.duration
                    }
                )
        else:
            if not HAS_PROMETHEUS:
                logger.warning(
                    "Prometheus 클라이언트 라이브러리가 설치되지 않았습니다.",
                    context={"feature": "prometheus_metrics"}
                )
            else:
                logger.info("설정에 따라 Prometheus 메트릭이 비활성화되었습니다.")

    def _setup_prometheus_metrics(self) -> None:
        """프로메테우스 메트릭 설정"""
        if not HAS_PROMETHEUS:
            return

        # 이미 메트릭이 초기화되었는지 확인하는 클래스 변수 추가
        if hasattr(self, '_metrics_initialized') and self._metrics_initialized:
            logger.info("Prometheus 메트릭이 이미 초기화되었습니다.", context={"action": "skip_registration"})
            return

        try:
            # 카운터 지표
            self.prometheus_metrics["total_requests"] = PromCounter(
                "vllm_total_requests", "Total number of requests processed"
            )
            self.prometheus_metrics["total_successes"] = PromCounter(
                "vllm_total_successes", "Total number of successful requests"
            )
            self.prometheus_metrics["total_failures"] = PromCounter(
                "vllm_total_failures", "Total number of failed requests"
            )
            self.prometheus_metrics["total_tokens_input"] = PromCounter(
                "vllm_total_tokens_input", "Total number of input tokens processed"
            )
            self.prometheus_metrics["total_tokens_output"] = PromCounter(
                "vllm_total_tokens_output", "Total number of output tokens generated"
            )

            # 게이지 지표
            self.prometheus_metrics["active_requests"] = Gauge(
                "vllm_active_requests", "Number of currently active requests"
            )

            # 히스토그램 지표
            self.prometheus_metrics["request_latency"] = Histogram(
                "vllm_request_latency_seconds", "Request latency in seconds",
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
            )
            self.prometheus_metrics["tokens_per_second"] = Histogram(
                "vllm_tokens_per_second", "Tokens generated per second",
                buckets=(1, 5, 10, 20, 50, 100, 200, 500)
            )

            # 시스템 메트릭
            self.prometheus_metrics["cpu_percent"] = Gauge(
                "vllm_cpu_percent", "CPU utilization percentage"
            )
            self.prometheus_metrics["ram_percent"] = Gauge(
                "vllm_ram_percent", "RAM utilization percentage"
            )
            self.prometheus_metrics["gpu_memory_percent"] = Gauge(
                "vllm_gpu_memory_percent", "GPU memory utilization percentage",
                ["gpu_id"]
            )
            self.prometheus_metrics["gpu_utilization"] = Gauge(
                "vllm_gpu_utilization", "GPU utilization percentage",
                ["gpu_id"]
            )

            # 성공적으로 초기화되었음을 표시
            self._metrics_initialized = True
            logger.info(
                "Prometheus 메트릭 초기화 완료",
                context={
                    "metrics_count": len(self.prometheus_metrics),
                    "metrics": list(self.prometheus_metrics.keys())
                }
            )

        except ValueError as e:
            # 중복 등록 오류 처리
            if "Duplicated timeseries" in str(e):
                logger.warning(
                    "Prometheus 메트릭이 이미 등록되어 있습니다.",
                    context={
                        "error": str(e),
                        "action": "using_existing_metrics"
                    }
                )
                self._metrics_initialized = True
                # 이미 등록된 메트릭 사용하기 위한 방법 (선택적)
                from prometheus_client import REGISTRY
                for metric in REGISTRY.collect():
                    if metric.name.startswith("vllm_"):
                        name = metric.name
                        if name.endswith("_total"):
                            name = name[:-6]  # "_total" 접미사 제거
                        if name not in self.prometheus_metrics:
                            self.prometheus_metrics[name] = metric
            else:
                # 다른 오류는 그대로 발생시킴
                logger.error(
                    "Prometheus 메트릭 초기화 중 오류 발생",
                    context={
                        "error_type": type(e).__name__,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise

    def start_collection(self) -> None:
        """메트릭 수집 스레드 시작"""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            logger.warning("메트릭 수집 스레드가 이미 실행 중입니다.")
            return

        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollectionThread"
        )
        self._collection_thread.start()
        logger.info("메트릭 수집 스레드가 시작되었습니다.", context={"thread_name": self._collection_thread.name})

    def stop_collection(self) -> None:
        """메트릭 수집 스레드 중지"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            logger.warning("메트릭 수집 스레드가 실행 중이 아닙니다.")
            return

        logger.info("메트릭 수집 스레드 중지 중...")
        self._stop_event.set()
        self._collection_thread.join(timeout=5.0)
        if self._collection_thread.is_alive():
            logger.warning(
                "메트릭 수집 스레드가 정상적으로 종료되지 않았습니다.",
                context={"thread_name": self._collection_thread.name}
            )
        else:
            logger.info("메트릭 수집 스레드가 정상적으로 종료되었습니다.")

    def _collection_loop(self) -> None:
        """메트릭 수집 루프"""
        last_log_time = time.time()
        collection_count = 0

        logger.info("메트릭 수집 루프 시작")

        while not self._stop_event.is_set():
            try:
                collection_count += 1
                # 시스템 메트릭 수집
                system_metrics = SystemMetrics.collect()
                with self.lock:
                    self.system_metrics_history.append(system_metrics)

                # 프로메테우스 시스템 메트릭 업데이트
                if self.enable_prometheus:
                    self._update_prometheus_system_metrics(system_metrics)

                # 상세 디버그 로깅 - 성능 모드에서는 빈도 감소
                log_frequency = 30 if performance_mode else 10
                if collection_count % log_frequency == 0 and not performance_mode:
                    logger.debug(
                        "시스템 메트릭 수집됨",
                        context={
                            "cpu_percent": system_metrics.cpu_percent,
                            "ram_percent": system_metrics.ram_percent,
                            "ram_used_gb": f"{system_metrics.ram_used_gb:.2f}",
                            "gpu_count": len(system_metrics.gpu_metrics)
                        }
                    )

                # 주기적 로깅
                current_time = time.time()
                if current_time - last_log_time >= self.log_interval:
                    self._log_metrics()
                    last_log_time = current_time

                # 완료된 요청 정리
                cleaned_count = self._cleanup_completed_requests()

                # 정리 결과 로깅 (성능 모드에서는 생략)
                if cleaned_count > 0 and not performance_mode:
                    logger.debug(
                        "완료된 요청 정리됨",
                        context={
                            "cleaned_count": cleaned_count,
                            "current_active": len(self.request_metrics),
                            "history_size": len(self.request_history)
                        }
                    )


                # 잠시 대기 (성능 모드에서는 더 긴 간격)
                sleep_time = 2.0 if performance_mode else 1.0
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(
                    "메트릭 수집 루프에서 오류 발생",
                    context={
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "collection_count": collection_count
                    },
                    exc_info=True
                )
                time.sleep(5.0)  # 오류 발생 시 좀 더 오래 대기

        logger.info("메트릭 수집 루프 종료")

    def _log_metrics(self) -> None:
        """현재 메트릭 정보 로깅"""
        with self.lock:
            if performance_mode:
                logger.info(
                    "성능 메트릭 요약",
                    context={
                        "metrics": {
                            "requests": {
                                "total": self.total_requests,
                                "active": len(self.request_metrics),
                            },
                            "tokens": {
                                "input": self.total_tokens_input,
                                "output": self.total_tokens_output
                            }
                        }
                    }
                )
                return

            # 성능 통계 계산
            avg_tokens_per_second = statistics.mean(self.tokens_per_second) if self.tokens_per_second else 0
            avg_request_time = statistics.mean(self.request_times) if self.request_times else 0

            # 시스템 메트릭 가져오기
            system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else SystemMetrics.collect()

            # GPU 정보 요약
            gpu_summary = []
            for gpu in system_metrics.gpu_metrics:
                gpu_summary.append({
                    "id": gpu["id"],
                    "memory_percent": f"{gpu.get('memory_used_percent', 0):.1f}%",
                    "utilization": f"{gpu.get('gpu_utilization', 0):.1f}%"
                })

            # 로그 데이터 구성
            logger.info(
                "성능 메트릭 요약",
                context={
                    "metrics": {
                        "uptime": f"{time.time() - self.start_time:.1f}초",
                        "requests": {
                            "total": self.total_requests,
                            "success": self.total_successes,
                            "failure": self.total_failures,
                            "active": len(self.request_metrics),
                        },
                        "tokens": {
                            "input": self.total_tokens_input,
                            "output": self.total_tokens_output,
                            "per_second_avg": f"{avg_tokens_per_second:.1f}"
                        },
                        "latency": {
                            "avg_request_time": f"{avg_request_time:.3f}초"
                        },
                        "system": {
                            "cpu": f"{system_metrics.cpu_percent:.1f}%",
                            "ram": f"{system_metrics.ram_percent:.1f}%",
                            "gpu": gpu_summary
                        }
                    }
                }
            )

            # 오류 통계 로깅 (있는 경우만)
            if self.error_counter:
                top_errors = dict(self.error_counter.most_common(5))
                logger.warning(
                    "최근 오류 통계",
                    context={
                        "top_errors": top_errors,
                        "total_errors": self.total_failures
                    }
                )

    def _cleanup_completed_requests(self) -> int:
        """
        오래된 완료된 요청 정리

        Returns:
            정리된 요청 수
        """
        with self.lock:
            current_time = time.time()
            to_remove = []

            for request_id, metrics in self.request_metrics.items():
                # 완료 또는 실패 상태이고 30초 이상 경과한 요청
                if (metrics.status in ("completed", "failed") and
                        metrics.end_time is not None and
                        current_time - metrics.end_time > 30.0):
                    to_remove.append(request_id)

            # 제거 및 히스토리에 추가
            for request_id in to_remove:
                metrics = self.request_metrics.pop(request_id)
                self.request_history.append(metrics.to_dict())

            return len(to_remove)  # 정리된 요청 수 반환

    def _update_prometheus_system_metrics(self, system_metrics: SystemMetrics) -> None:
        """프로메테우스 시스템 메트릭 업데이트"""
        if not self.enable_prometheus:
            return

        # CPU 및 RAM 메트릭
        self.prometheus_metrics["cpu_percent"].set(system_metrics.cpu_percent)
        self.prometheus_metrics["ram_percent"].set(system_metrics.ram_percent)

        # GPU 메트릭
        for gpu in system_metrics.gpu_metrics:
            gpu_id = str(gpu["id"])
            if "memory_used_percent" in gpu:
                self.prometheus_metrics["gpu_memory_percent"].labels(gpu_id=gpu_id).set(gpu["memory_used_percent"])
            if "gpu_utilization" in gpu:
                self.prometheus_metrics["gpu_utilization"].labels(gpu_id=gpu_id).set(gpu["gpu_utilization"])

    def start_request(self, request_id: str, prompt_tokens: int = 0, client_ip: Optional[str] = None) -> None:
        """
        요청 처리 시작 기록

        Args:
            request_id: 요청 ID
            prompt_tokens: 프롬프트 토큰 수
            client_ip: 클라이언트 IP 주소
        """
        with self.lock:
            self.total_requests += 1
            if self.enable_prometheus:
                self.prometheus_metrics["total_requests"].inc()
                self.prometheus_metrics["active_requests"].inc()

            # 요청 메트릭 생성
            metrics = RequestMetrics(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                client_ip=client_ip
            )
            self.request_metrics[request_id] = metrics
            self.total_tokens_input += prompt_tokens

            if self.enable_prometheus:
                self.prometheus_metrics["total_tokens_input"].inc(prompt_tokens)

            # 디버그 로깅
            if not performance_mode:
                logger.debug(
                    "요청 시작됨",
                    context={
                        "request_id": request_id,
                        "prompt_tokens": prompt_tokens,
                        "client_ip": client_ip,
                        "active_requests": len(self.request_metrics)
                    }
                )

    def start_processing(self, request_id: str, prompt_tokens: Optional[int] = None) -> None:
        """
        요청 처리 시작 (큐에서 꺼내어 실제 처리 시작)

        Args:
            request_id: 요청 ID
            prompt_tokens: 프롬프트 토큰 수 (알려진 경우)
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(
                    "요청 메트릭을 찾을 수 없음",
                    context={
                        "request_id": request_id,
                        "action": "start_processing",
                        "available_requests": len(self.request_metrics)
                    }
                )
                return

            metrics = self.request_metrics[request_id]
            current_time = time.time()
            metrics.queue_time = current_time - metrics.start_time
            metrics.status = "processing"

            if prompt_tokens is not None:
                # 이전에 설정된 값과 다르면 차이만큼 total_tokens_input 업데이트
                token_diff = prompt_tokens - metrics.prompt_tokens
                self.total_tokens_input += token_diff
                metrics.prompt_tokens = prompt_tokens

                if self.enable_prometheus and token_diff != 0:
                    self.prometheus_metrics["total_tokens_input"].inc(token_diff)

            # 디버그 로깅
            logger.debug(
                "요청 처리 시작됨",
                context={
                    "request_id": request_id,
                    "queue_time": f"{metrics.queue_time:.3f}초",
                    "prompt_tokens": metrics.prompt_tokens
                }
            )

    def complete_request(self, request_id: str, completion_tokens: int,
                         prompt_tokens: Optional[int] = None) -> None:
        """
        요청 처리 완료 기록

        Args:
            request_id: 요청 ID
            completion_tokens: 생성된 토큰 수
            prompt_tokens: 프롬프트 토큰 수 (알려진 경우)
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(
                    "요청 메트릭을 찾을 수 없음",
                    context={
                        "request_id": request_id,
                        "action": "complete_request",
                        "completion_tokens": completion_tokens
                    }
                )
                return

            metrics = self.request_metrics[request_id]
            completion_info = metrics.complete(completion_tokens, prompt_tokens)

            # 통계 업데이트
            self.total_successes += 1
            self.total_tokens_output += completion_tokens
            self.request_times.append(metrics.total_time)
            self.tokens_per_second.append(metrics.tokens_per_second)

            # 프로메테우스 메트릭 업데이트
            if self.enable_prometheus:
                self.prometheus_metrics["total_successes"].inc()
                self.prometheus_metrics["total_tokens_output"].inc(completion_tokens)
                self.prometheus_metrics["request_latency"].observe(metrics.total_time)
                self.prometheus_metrics["tokens_per_second"].observe(metrics.tokens_per_second)
                self.prometheus_metrics["active_requests"].dec()

        # 성능 모드 여부에 따라 로깅 조정
        if performance_mode:
            # 성능 모드: 느린 요청만 상세 로깅 (>1초)
            if metrics.total_time > 1.0:
                logger.info(
                    "요청 처리 완료 (느린 요청)",
                    context={
                        "request_id": request_id,
                        "metrics": {
                            "completion_tokens": completion_tokens,
                            "total_time": f"{metrics.total_time:.3f}초",
                            "tokens_per_second": f"{metrics.tokens_per_second:.2f}"
                        }
                    }
                )
        else:
            logger.info(
                    "요청 처리 완료",
                    context={
                        "request_id": request_id,
                        "metrics": {
                            "completion_tokens": completion_tokens,
                            "prompt_tokens": metrics.prompt_tokens,
                            "total_tokens": metrics.total_tokens,
                            "total_time": f"{metrics.total_time:.3f}초",
                            "tokens_per_second": f"{metrics.tokens_per_second:.2f}",
                            "status": "completed"
                        }
                    }
                )

    def fail_request(self, request_id: str, error: str) -> None:
        """
        요청 처리 실패 기록

        Args:
            request_id: 요청 ID
            error: 오류 메시지
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(
                    "요청 메트릭을 찾을 수 없음",
                    context={
                        "request_id": request_id,
                        "action": "fail_request",
                        "error_message": error[:100] + "..." if len(error) > 100 else error
                    }
                )
                return

            metrics = self.request_metrics[request_id]
            fail_info = metrics.fail(error)

            # 통계 업데이트
            self.total_failures += 1
            self.error_counter[error] += 1

            # 프로메테우스 메트릭 업데이트
            if self.enable_prometheus:
                self.prometheus_metrics["total_failures"].inc()
                self.prometheus_metrics["active_requests"].dec()

            # 오류 로깅
            logger.error(
                "요청 처리 실패",
                context={
                    "request_id": request_id,
                    "metrics": {
                        "total_time": f"{metrics.total_time:.3f}초",
                        "status": "failed"
                    },
                    "error": {
                        "message": error,
                        "count": self.error_counter[error]
                    }
                }
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        현재 메트릭 정보 반환

        Returns:
            메트릭 정보 딕셔너리
        """
        with self.lock:
            # 성능 통계 계산
            avg_tokens_per_second = statistics.mean(self.tokens_per_second) if self.tokens_per_second else 0
            avg_request_time = statistics.mean(self.request_times) if self.request_times else 0

            # 시스템 메트릭 가져오기
            system_metrics = self.system_metrics_history[
                -1] if self.system_metrics_history else SystemMetrics.collect()

            # 활성 요청 정보
            active_requests = {
                req_id: metrics.to_dict()
                for req_id, metrics in self.request_metrics.items()
            }

            # 상위 오류 목록 (최대 10개)
            top_errors = dict(self.error_counter.most_common(10))

            # 로깅
            logger.debug(
                "메트릭 요청됨",
                context={
                    "active_requests": len(self.request_metrics),
                    "total_requests": self.total_requests
                }
            )

            # 결과 구성
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.start_time,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "active_requests_count": len(self.request_metrics),
                "active_requests": active_requests,
                "total_tokens_input": self.total_tokens_input,
                "total_tokens_output": self.total_tokens_output,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_request_time": avg_request_time,
                "error_stats": top_errors,
                "system": asdict(system_metrics)
            }

    def get_request_metrics(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 요청의 메트릭 정보 반환

        Args:
            request_id: 요청 ID

        Returns:
            요청 메트릭 정보 딕셔너리, 없으면 None
        """
        with self.lock:
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id].to_dict()
                logger.debug(
                    "활성 요청 메트릭 반환",
                    context={
                        "request_id": request_id,
                        "status": metrics["status"]
                    }
                )
                return metrics

            # 히스토리에서 찾기
            for metrics in self.request_history:
                if metrics["request_id"] == request_id:
                    logger.debug(
                        "히스토리에서 요청 메트릭 찾음",
                        context={
                            "request_id": request_id,
                            "status": metrics["status"]
                        }
                    )
                    return metrics

            logger.warning(
                "요청 메트릭을 찾을 수 없음",
                context={
                    "request_id": request_id,
                    "active_count": len(self.request_metrics),
                    "history_count": len(self.request_history)
                }
            )
            return None


# 싱글톤 메트릭 수집기
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """
    메트릭 수집기 인스턴스 가져오기 (싱글톤)

    Returns:
        MetricsCollector 인스턴스
    """
    global _metrics_collector

    # 모니터링이 비활성화된 경우 가벼운 더미 컬렉터 반환 고려
    if not config.monitoring.enabled:
        logger.debug("모니터링이 비활성화되어 더미 컬렉터 반환")
        return _create_dummy_collector()

    # 성능 모드에서는 이미 생성된 인스턴스가 있으면 추가 로깅 없이 반환
    if _metrics_collector is not None and performance_mode:
        return _metrics_collector

    if _metrics_collector is None:
        # 성능 모드에서는 로깅 간격 늘리기
        log_interval = config.monitoring.log_stats_interval
        if performance_mode:
            log_interval = max(30, log_interval * 2)  # 최소 30초, 기본값의 2배

        with TimingContext(logger, "메트릭 수집기 초기화",
                           log_threshold=0.1 if performance_mode else None) as timing:
            _metrics_collector = MetricsCollector(
                max_history=config.monitoring.log_stats_interval * (5 if performance_mode else 10),
                enable_prometheus=config.monitoring.prometheus,
                prometheus_port=config.monitoring.metrics_port,
                log_interval=log_interval
            )
            _metrics_collector.start_collection()

            if performance_mode:
                logger.info("메트릭 수집기 초기화 완료")
            else:
                logger.info(
                    "메트릭 수집기 초기화 완료",
                    context={
                        "init_time": f"{timing.duration:.3f}초",
                        "prometheus_enabled": config.monitoring.prometheus,
                        "max_history": config.monitoring.log_stats_interval * 10
                    }
                )

    return _metrics_collector

    # 더미 컬렉터 생성 함수 추가


def _create_dummy_collector():
    """모니터링 비활성화 상태에서 사용할 더미 컬렉터 생성"""
    if not hasattr(get_metrics_collector, '_dummy_collector'):
        class DummyCollector:
            """모니터링이 비활성화된 경우 사용되는 더미 컬렉터 클래스"""

            def __init__(self):
                logger.debug("더미 메트릭 컬렉터 생성됨")

            def start_request(self, *args, **kwargs):
                logger.debug("더미 컬렉터: start_request 호출됨 (무시됨)")
                pass

            def start_processing(self, *args, **kwargs):
                logger.debug("더미 컬렉터: start_processing 호출됨 (무시됨)")
                pass

            def complete_request(self, *args, **kwargs):
                logger.debug("더미 컬렉터: complete_request 호출됨 (무시됨)")
                pass

            def fail_request(self, *args, **kwargs):
                logger.debug("더미 컬렉터: fail_request 호출됨 (무시됨)")
                pass

            def get_metrics(self):
                logger.debug("더미 컬렉터: get_metrics 호출됨")
                return {"status": "monitoring_disabled"}

            def get_request_metrics(self, *args):
                logger.debug("더미 컬렉터: get_request_metrics 호출됨")
                return None

            def start_collection(self):
                logger.debug("더미 컬렉터: start_collection 호출됨 (무시됨)")
                pass

            def stop_collection(self):
                logger.debug("더미 컬렉터: stop_collection 호출됨 (무시됨)")
                pass

        get_metrics_collector._dummy_collector = DummyCollector()
        logger.info("더미 메트릭 컬렉터가 생성되었습니다 (모니터링 비활성화됨)")

    return get_metrics_collector._dummy_collector


@with_logging_context
def track_request(func):
    """
    요청 메트릭 추적 데코레이터

    요청 시작/완료/실패를 자동으로 추적하는 데코레이터

    Args:
        func: 원본 함수

    Returns:
        래핑된 함수
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 성능 모드 감지
        is_perf_mode = performance_mode

        # 모니터링이 비활성화된 경우 원본 함수만 실행
        if not config.monitoring.enabled:
            if not is_perf_mode:
                logger.debug("모니터링이 비활성화되어 메트릭 추적 없이 함수 실행")
            return await func(*args, **kwargs)

        # 요청 ID 추출 또는 생성
        request_id = None

        # 요청 객체에서 request_id 추출 시도
        if 'request_id' in kwargs:
            request_id = kwargs['request_id']
        else:
            # 요청 객체에서 request_id 추출 시도
            for arg in args:
                if hasattr(arg, 'request_id') and arg.request_id:
                    request_id = arg.request_id
                    break

        # 생성되지 않은 경우 새로 생성
        if request_id is None:
            request_id = str(uuid.uuid4())

        # 요청 데이터 추출
        client_ip = None
        prompt_tokens = 0

        if not is_perf_mode:
            for arg in args:
                if hasattr(arg, 'client') and hasattr(arg.client, 'host'):
                    client_ip = arg.client.host
                if hasattr(arg, 'prompt'):
                    # 프롬프트 토큰 수 추출 시도
                    if hasattr(arg, 'prompt_tokens'):
                        prompt_tokens = arg.prompt_tokens

        # 컨텍스트에 요청 ID 추가하여 로깅
        logger.with_context(request_id=request_id)

        if not is_perf_mode:
            logger.info(
                "API 요청 처리 시작",
                context={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "function": func.__name__
                }
            )

        # 요청 시작 기록
        collector = get_metrics_collector()
        collector.start_request(request_id, prompt_tokens=prompt_tokens, client_ip=client_ip)

        # 타이밍 측정
        start_time = time.time()

        try:
            # 요청 처리 시작
            collector.start_processing(request_id)

            # 원본 함수 호출
            with TimingContext(logger, f"API 함수 {func.__name__}") as timing:
                result = await func(*args, **kwargs)

            # 실행 시간 계산
            execution_time = timing.duration

            # 요청 완료 시 메트릭 업데이트
            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)

                collector.complete_request(
                    request_id,
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens
                )

                if not is_perf_mode or execution_time > 0.5:
                    logger.info(
                        "API 요청 처리 완료",
                        context={
                            "request_id": request_id,
                            "execution_time": f"{execution_time:.3f}초",
                            "tokens": {
                                "prompt": prompt_tokens,
                                "completion": completion_tokens,
                                "total": prompt_tokens + completion_tokens
                            }
                        }
                    )
            else:
                # 사용량 정보가 없는 경우
                collector.complete_request(request_id, completion_tokens=0)

                if not is_perf_mode or execution_time > 0.5:
                    logger.info(
                        "API 요청 처리 완료 (토큰 정보 없음)",
                        context={
                            "request_id": request_id,
                            "execution_time": f"{execution_time:.3f}초"
                        }
                    )

            return result

        except Exception as e:
            # 오류 발생 시 실패 기록
            execution_time = time.time() - start_time
            collector.fail_request(request_id, str(e))

            logger.error(
                "API 요청 처리 실패",
                context={
                    "request_id": request_id,
                    "execution_time": f"{execution_time:.3f}초",
                    "error_type": type(e).__name__,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
        finally:
            # 컨텍스트 정리
            logger.clear_context()

    return wrapper


async def monitor_gpu_memory():
    """GPU 메모리 사용량 모니터링"""
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA를 사용할 수 없습니다. GPU 모니터링이 비활성화됩니다.",
            context={"feature": "gpu_monitoring"}
        )
        return

    # 성능 모드에서는 모니터링 간격 증가
    monitor_interval = 30 if performance_mode else 10  # 초

    try:
        # GPU 개수 확인
        num_gpus = torch.cuda.device_count()
        logger.info(
            "GPU 모니터링 시작",
            context={
                "gpu_count": num_gpus,
                "interval": f"{monitor_interval}초"
            }
        )

        last_alert_time = 0  # 마지막 경고 시간
        alert_threshold = 0.9  # 경고 임계값 (90%)

        while True:
            gpu_stats = []
            high_usage_detected = False

            for i in range(num_gpus):
                try:
                    # 메모리 사용량 (GB)
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9

                    # GPU 정보 가져오기
                    gpu_name = torch.cuda.get_device_name(i)

                    # 사용률 계산 (예상치)
                    if HAS_GPUTIL:
                        try:
                            gpus = GPUtil.getGPUs()
                            if i < len(gpus):
                                usage_percent = gpus[i].memoryUtil * 100
                                temp = gpus[i].temperature
                            else:
                                usage_percent = (allocated / reserved) * 100 if reserved > 0 else 0
                                temp = None
                        except Exception:
                            usage_percent = (allocated / reserved) * 100 if reserved > 0 else 0
                            temp = None
                    else:
                        usage_percent = (allocated / reserved) * 100 if reserved > 0 else 0
                        temp = None

                    gpu_info = {
                        "id": i,
                        "name": gpu_name,
                        "allocated_gb": f"{allocated:.2f}",
                        "reserved_gb": f"{reserved:.2f}",
                        "usage_percent": f"{usage_percent:.1f}%"
                    }

                    if temp is not None:
                        gpu_info["temperature"] = f"{temp}°C"

                    gpu_stats.append(gpu_info)

                    # 높은 사용률 감지
                    if usage_percent > alert_threshold * 100:
                        high_usage_detected = True

                except Exception as gpu_err:
                    logger.warning(
                        f"GPU {i} 모니터링 중 오류 발생",
                        context={
                            "gpu_id": i,
                            "error": str(gpu_err)
                        }
                    )

            if not performance_mode:
                logger.debug(
                    "GPU 메모리 상태",
                    context={
                        "gpus": gpu_stats
                    }
                )

            # 높은 사용률 감지 시 경고 (최소 5분에 한 번)
            current_time = time.time()
            if high_usage_detected and (current_time - last_alert_time) > 300:
                logger.warning(
                    "높은 GPU 메모리 사용률 감지됨",
                    context={
                        "gpus": gpu_stats,
                        "threshold": f"{alert_threshold * 100}%"
                    }
                )
                last_alert_time = current_time

            # 주기적 체크
            await asyncio.sleep(monitor_interval)

    except Exception as e:
        logger.error(
            "GPU 모니터링 중 오류 발생",
            context={
                "error_type": type(e).__name__,
                "error": str(e)
            },
            exc_info=True
        )


# 모듈 초기화
def init_monitoring():
    """
    모니터링 모듈 초기화

    서버 시작 시 호출됨
    """
    if config.monitoring.enabled:
        try:
            # 메트릭 수집기 초기화
            collector = get_metrics_collector()
            logger.info(
                "모니터링 시스템 초기화됨",
                context={
                    "prometheus_enabled": config.monitoring.prometheus,
                    "metrics_port": config.monitoring.metrics_port,
                    "log_interval": config.monitoring.log_stats_interval,
                    "record_memory": config.monitoring.record_memory
                }
            )

            # GPU 메모리 모니터링 시작 (비동기 함수이므로 직접 호출하지 않음)
            if config.monitoring.record_memory and torch.cuda.is_available():
                logger.info("GPU 메모리 모니터링이 활성화되었습니다")

            return collector
        except Exception as e:
            logger.error(
                "모니터링 초기화 실패",
                context={
                    "error_type": type(e).__name__,
                    "error": str(e)
                },
                exc_info=True
            )
            return None
    else:
        logger.info("설정에 따라 모니터링이 비활성화되었습니다")
        return None


# 메트릭 API 엔드포인트 핸들러
async def get_current_metrics():
    """
    현재 메트릭 정보 가져오기

    Returns:
        현재 메트릭 정보 딕셔너리
    """
    try:
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        logger.debug("현재 메트릭 정보가 요청되었습니다")
        return metrics
    except Exception as e:
        logger.error(
            "메트릭 정보 조회 중 오류 발생",
            context={
                "error_type": type(e).__name__,
                "error": str(e)
            },
            exc_info=True
        )
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def get_request_metrics_by_id(request_id: str):
    """
    특정 요청의 메트릭 정보 가져오기

    Args:
        request_id: 요청 ID

    Returns:
        요청 메트릭 정보 딕셔너리, 없으면 None
    """
    try:
        collector = get_metrics_collector()
        metrics = collector.get_request_metrics(request_id)
        if metrics is None:
            logger.warning(
                "요청된 메트릭을 찾을 수 없음",
                context={
                    "request_id": request_id
                }
            )
        else:
            logger.debug(
                "요청 메트릭 정보가 조회됨",
                context={
                    "request_id": request_id,
                    "status": metrics.get("status", "unknown")
                }
            )
        return metrics
    except Exception as e:
        logger.error(
            "요청 메트릭 정보 조회 중 오류 발생",
            context={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error": str(e)
            },
            exc_info=True
        )
        return {
            "status": "error",
            "error": str(e),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
