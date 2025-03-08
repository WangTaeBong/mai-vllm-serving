"""
mai-vllm-serving의 모니터링 지표 모듈
성능, 리소스 사용량, 요청 처리 등에 관한 지표를 수집하고 관리
"""

import asyncio
import functools
import json
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
from mai_vllm_serving.utils.logging_utils import setup_logging

# 설정 객체 가져오기
config = get_config()

# 로깅 초기화
logger = setup_logging(
    service_name="mai-vllm-serving-metrics",
    log_level=config.logging.level,
    use_json=config.logging.json,
    log_file=config.logging.file
)


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

    def fail(self, error: str) -> None:
        """요청 실패 처리"""
        self.end_time = time.time()
        self.error = error
        self.status = "failed"


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
                logger.warning(f"Failed to collect GPU metrics: {str(e)}")

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
                    logger.warning(f"Failed to collect PyTorch CUDA metrics for GPU {i}: {str(e)}")

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
        self.request_history: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self.system_metrics_history: Deque[SystemMetrics] = deque(maxlen=max_history)
        self.start_time = time.time()

        # 통계 정보
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0

        # 성능 통계
        self.request_times: Deque[float] = deque(maxlen=max_history)
        self.tokens_per_second: Deque[float] = deque(maxlen=max_history)

        # 오류 통계
        self.error_counter = Counter()

        # 스레드 잠금
        self.lock = threading.RLock()

        # 프로메테우스 지표 (사용 가능한 경우)
        self.enable_prometheus = enable_prometheus and HAS_PROMETHEUS
        self.prometheus_port = prometheus_port
        self.prometheus_metrics = {}

        # 주기적 작업 스레드
        self.log_interval = log_interval
        self._stop_event = threading.Event()
        self._collection_thread = None

        # 메트릭 초기화 상태 추적을 위한 변수 추가
        self._metrics_initialized = False

        # 프로메테우스 설정
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
            start_prometheus_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        else:
            if not HAS_PROMETHEUS:
                logger.warning("Prometheus client library not installed. Prometheus metrics disabled.")
            else:
                logger.info("Prometheus metrics disabled in configuration.")

    def _setup_prometheus_metrics(self) -> None:
        """프로메테우스 메트릭 설정"""
        if not HAS_PROMETHEUS:
            return

        # 이미 메트릭이 초기화되었는지 확인하는 클래스 변수 추가
        if hasattr(self, '_metrics_initialized') and self._metrics_initialized:
            logger.info("Prometheus metrics already initialized, skipping registration")
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
            logger.info("Prometheus metrics successfully initialized")

        except ValueError as e:
            # 중복 등록 오류 처리
            if "Duplicated timeseries" in str(e):
                logger.warning(f"Prometheus metrics already registered: {str(e)}")
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
                logger.error(f"Error initializing Prometheus metrics: {str(e)}")
                raise

    def start_collection(self) -> None:
        """메트릭 수집 스레드 시작"""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            logger.warning("Metrics collection thread is already running")
            return

        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollectionThread"
        )
        self._collection_thread.start()
        logger.info("Metrics collection thread started")

    def stop_collection(self) -> None:
        """메트릭 수집 스레드 중지"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            logger.warning("Metrics collection thread is not running")
            return

        self._stop_event.set()
        self._collection_thread.join(timeout=5.0)
        if self._collection_thread.is_alive():
            logger.warning("Metrics collection thread did not terminate properly")
        else:
            logger.info("Metrics collection thread stopped")

    def _collection_loop(self) -> None:
        """메트릭 수집 루프"""
        last_log_time = time.time()

        while not self._stop_event.is_set():
            try:
                # 시스템 메트릭 수집
                system_metrics = SystemMetrics.collect()
                with self.lock:
                    self.system_metrics_history.append(system_metrics)

                # 프로메테우스 시스템 메트릭 업데이트
                if self.enable_prometheus:
                    self._update_prometheus_system_metrics(system_metrics)

                # 주기적 로깅
                current_time = time.time()
                if current_time - last_log_time >= self.log_interval:
                    self._log_metrics()
                    last_log_time = current_time

                # 완료된 요청 정리
                self._cleanup_completed_requests()

                # 잠시 대기
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}", exc_info=True)
                time.sleep(5.0)  # 오류 발생 시 좀 더 오래 대기

    def _log_metrics(self) -> None:
        """현재 메트릭 정보 로깅"""
        with self.lock:
            # 성능 통계 계산
            avg_tokens_per_second = statistics.mean(self.tokens_per_second) if self.tokens_per_second else 0
            avg_request_time = statistics.mean(self.request_times) if self.request_times else 0

            # 시스템 메트릭 가져오기
            system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else SystemMetrics.collect()

            # 로그 데이터 구성
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "uptime": time.time() - self.start_time,
                "total_requests": self.total_requests,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "active_requests": len(self.request_metrics),
                "total_tokens_input": self.total_tokens_input,
                "total_tokens_output": self.total_tokens_output,
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_request_time": avg_request_time,
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "ram_percent": system_metrics.ram_percent,
                    "gpu_metrics": system_metrics.gpu_metrics,
                    "disk_percent": system_metrics.disk_percent
                }
            }

            logger.info(f"Performance metrics: {json.dumps(log_data)}")

    def _cleanup_completed_requests(self) -> None:
        """오래된 완료된 요청 정리"""
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

    def start_processing(self, request_id: str, prompt_tokens: Optional[int] = None) -> None:
        """
        요청 처리 시작 (큐에서 꺼내어 실제 처리 시작)

        Args:
            request_id: 요청 ID
            prompt_tokens: 프롬프트 토큰 수 (알려진 경우)
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(f"Request {request_id} not found in metrics")
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

    def complete_request(self, request_id: str, completion_tokens: int, prompt_tokens: Optional[int] = None) -> None:
        """
        요청 처리 완료 기록

        Args:
            request_id: 요청 ID
            completion_tokens: 생성된 토큰 수
            prompt_tokens: 프롬프트 토큰 수 (알려진 경우)
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(f"Request {request_id} not found in metrics")
                return

            metrics = self.request_metrics[request_id]
            metrics.complete(completion_tokens, prompt_tokens)

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

    def fail_request(self, request_id: str, error: str) -> None:
        """
        요청 처리 실패 기록

        Args:
            request_id: 요청 ID
            error: 오류 메시지
        """
        with self.lock:
            if request_id not in self.request_metrics:
                logger.warning(f"Request {request_id} not found in metrics")
                return

            metrics = self.request_metrics[request_id]
            metrics.fail(error)

            # 통계 업데이트
            self.total_failures += 1
            self.error_counter[error] += 1

            # 프로메테우스 메트릭 업데이트
            if self.enable_prometheus:
                self.prometheus_metrics["total_failures"].inc()
                self.prometheus_metrics["active_requests"].dec()

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
            system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else SystemMetrics.collect()

            # 활성 요청 정보
            active_requests = {
                req_id: metrics.to_dict()
                for req_id, metrics in self.request_metrics.items()
            }

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
                "error_stats": dict(self.error_counter),
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
                return self.request_metrics[request_id].to_dict()

            # 히스토리에서 찾기
            for metrics in self.request_history:
                if metrics["request_id"] == request_id:
                    return metrics

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
        return _create_dummy_collector()

    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(
            max_history=config.monitoring.log_stats_interval * 10,
            enable_prometheus=config.monitoring.prometheus,
            prometheus_port=config.monitoring.metrics_port,
            log_interval=config.monitoring.log_stats_interval
        )
        _metrics_collector.start_collection()

    return _metrics_collector


# 더미 컬렉터 생성 함수 추가
def _create_dummy_collector():
    """모니터링 비활성화 상태에서 사용할 더미 컬렉터 생성"""
    global _dummy_collector

    if not hasattr(get_metrics_collector, '_dummy_collector'):
        class DummyCollector:
            def start_request(self, *args, **kwargs): pass

            def start_processing(self, *args, **kwargs): pass

            def complete_request(self, *args, **kwargs): pass

            def fail_request(self, *args, **kwargs): pass

            def get_metrics(self): return {"status": "monitoring_disabled"}

            def get_request_metrics(self, *args): return None

            def start_collection(self): pass

            def stop_collection(self): pass

        get_metrics_collector._dummy_collector = DummyCollector()

    return get_metrics_collector._dummy_collector


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
        # 모니터링이 비활성화된 경우 원본 함수만 실행
        if not config.monitoring.enabled:
            return await func(*args, **kwargs)

        # 요청 ID 생성
        request_id = str(uuid.uuid4())

        # 요청 데이터 추출
        request_data = None
        client_ip = None
        for arg in args:
            if hasattr(arg, 'client') and hasattr(arg.client, 'host'):
                client_ip = arg.client.host
            if hasattr(arg, 'prompt'):
                request_data = arg

        # 클라이언트 IP 추출 (키워드 인자에서)
        if client_ip is None and 'client_request' in kwargs:
            client_ip = kwargs['client_request'].client.host if hasattr(kwargs['client_request'], 'client') else None

        # 요청 시작 기록
        collector = get_metrics_collector()
        collector.start_request(request_id, client_ip=client_ip)

        try:
            # 요청 처리 시작
            collector.start_processing(request_id)

            # 원본 함수 호출
            result = await func(*args, **kwargs)

            # 요청 완료 시 메트릭 업데이트
            if isinstance(result, dict) and "usage" in result:
                usage = result["usage"]
                collector.complete_request(
                    request_id,
                    completion_tokens=usage.get("completion_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0)
                )
            else:
                # 사용량 정보가 없는 경우
                collector.complete_request(request_id, completion_tokens=0)

            return result

        except Exception as e:
            # 오류 발생 시 실패 기록
            collector.fail_request(request_id, str(e))
            raise

    return wrapper


async def monitor_gpu_memory():
    """GPU 메모리 사용량 모니터링"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. GPU monitoring disabled.")
        return

    try:
        # GPU 개수 확인
        num_gpus = torch.cuda.device_count()
        logger.info(f"Monitoring {num_gpus} GPUs")

        while True:
            for i in range(num_gpus):
                # 메모리 사용량 (GB)
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9

                # 로깅
                logger.debug(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

            # 주기적 체크 (10초마다)
            await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"Error in GPU monitoring: {str(e)}", exc_info=True)


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
            logger.info("Metrics collector initialized")

            # GPU 메모리 모니터링 시작 (비동기 함수이므로 직접 호출하지 않음)
            if config.monitoring.record_memory and torch.cuda.is_available():
                logger.info("GPU memory monitoring enabled")

            return collector
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {str(e)}", exc_info=True)
            return None
    else:
        logger.info("Monitoring is disabled in configuration")
        return None


# 메트릭 API 엔드포인트 핸들러
async def get_current_metrics():
    """
    현재 메트릭 정보 가져오기

    Returns:
        현재 메트릭 정보 딕셔너리
    """
    collector = get_metrics_collector()
    return collector.get_metrics()


async def get_request_metrics_by_id(request_id: str):
    """
    특정 요청의 메트릭 정보 가져오기

    Args:
        request_id: 요청 ID

    Returns:
        요청 메트릭 정보 딕셔너리, 없으면 None
    """
    collector = get_metrics_collector()
    return collector.get_request_metrics(request_id)
