"""
mai-vllm-serving 성능 프로파일링 모듈
시스템 성능 측정 및 분석을 위한 상세 프로파일링 기능 제공
"""
import asyncio
import contextlib
import functools
import gc
import json
import logging
import threading
import time
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, cast

import psutil
import torch

try:
    import torch.profiler as torch_profiler

    HAS_TORCH_PROFILER = True
except ImportError:
    torch_profiler = None
    HAS_TORCH_PROFILER = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import py3nvml
    import py3nvml.py3nvml as nvml

    HAS_NVML = True
except ImportError:
    nvml = None
    HAS_NVML = False

# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config
from mai_vllm_serving.utils.logging_utils import setup_logging

# 로깅 설정
config = get_config()
logger = setup_logging(
    service_name="mai-vllm-serving-profiler",
    log_level=config.logging.level,
    log_file=config.logging.file
)


@dataclass
class FunctionProfile:
    """함수 호출 프로파일 정보"""
    function_name: str
    module_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: float = 0.0

    def update(self, execution_time: float) -> None:
        """
        함수 실행 시간 정보 업데이트

        Args:
            execution_time: 함수 실행 시간 (초)
        """
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.last_call_time = execution_time


@dataclass
class MemorySnapshot:
    """메모리 사용 스냅샷"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_ram_used_gb: float = 0.0
    system_ram_total_gb: float = 0.0
    system_ram_percent: float = 0.0
    process_ram_used_gb: float = 0.0
    python_objects: Dict[str, int] = field(default_factory=dict)
    gpu_memory: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def capture() -> 'MemorySnapshot':
        """
        현재 메모리 사용 상태 캡처

        Returns:
            메모리 스냅샷 객체
        """
        snapshot = MemorySnapshot()

        # 시스템 RAM 정보
        mem = psutil.virtual_memory()
        snapshot.system_ram_used_gb = mem.used / (1024 ** 3)
        snapshot.system_ram_total_gb = mem.total / (1024 ** 3)
        snapshot.system_ram_percent = mem.percent

        # 현재 프로세스 메모리 사용량
        pid = os.getpid()
        process = psutil.Process(pid)
        snapshot.process_ram_used_gb = process.memory_info().rss / (1024 ** 3)

        # Python 객체 카운트
        if HAS_MEMORY_PROFILER:
            objects = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                if obj_type in objects:
                    objects[obj_type] += 1
                else:
                    objects[obj_type] = 1

            # 가장 많은 객체 타입 상위 20개만 저장
            snapshot.python_objects = dict(
                sorted(objects.items(), key=lambda x: x[1], reverse=True)[:20]
            )

        # GPU 메모리 정보
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                gpu_info = {
                    "id": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "memory_allocated_gb": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3),
                    "memory_cached_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3) if hasattr(torch.cuda,
                                                                                                 'memory_reserved') else 0
                }

                if HAS_NVML:
                    try:
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                        info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info["total_memory_gb"] = info.total / (1024 ** 3)
                        gpu_info["used_memory_gb"] = info.used / (1024 ** 3)
                        gpu_info["free_memory_gb"] = info.free / (1024 ** 3)

                        # GPU 사용률
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_info["gpu_utilization"] = util.gpu
                        gpu_info["memory_utilization"] = util.memory

                        # 온도
                        gpu_info["temperature"] = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        nvml.nvmlShutdown()
                    except Exception as err:
                        logger.debug(f"NVML error: {str(err)}")

                snapshot.gpu_memory.append(gpu_info)

        return snapshot


@dataclass
class ProfilerConfig:
    """프로파일러 설정"""
    enabled: bool = True
    profile_interval: int = 60  # 주기적 프로파일링 간격 (초)
    memory_profiling: bool = True  # 메모리 프로파일링 활성화 여부
    torch_profiling: bool = True  # PyTorch 프로파일링 활성화 여부
    log_to_file: bool = True  # 파일에 프로파일 결과 기록 여부
    profile_dir: str = "./profiles"  # 프로파일 결과 저장 디렉토리
    verbose: bool = False  # 상세 로깅 여부


class SystemProfiler:
    """
    시스템 성능 프로파일링 클래스

    시스템, 프로세스, GPU 등의 성능 측정 및 분석
    """

    def __init__(self, p_config: Optional[ProfilerConfig] = None):
        """
        시스템 프로파일러 초기화

        Args:
            p_config: 프로파일러 설정
        """
        self.config = p_config or ProfilerConfig()

        if not self.config.enabled:
            logger.info("Profiler is disabled")
            return

        # 함수 프로파일 저장소
        self.function_profiles: Dict[str, FunctionProfile] = {}

        # 메모리 스냅샷 이력
        self.memory_snapshots: List[MemorySnapshot] = []

        # 프로파일 결과 저장 디렉토리 생성
        if self.config.log_to_file:
            os.makedirs(self.config.profile_dir, exist_ok=True)

        # 주기적 프로파일링 스레드
        self._stop_event = threading.Event()
        self._profiling_thread = None

        # PyTorch 프로파일링 세션
        self.torch_profiler_session = None

        logger.info(f"System profiler initialized with interval {self.config.profile_interval}s")

    def start(self) -> None:
        """주기적 프로파일링 시작"""
        if not self.config.enabled:
            return

        if self._profiling_thread is not None and self._profiling_thread.is_alive():
            logger.warning("Profiling thread is already running")
            return

        self._stop_event.clear()
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True,
            name="ProfilingThread"
        )
        self._profiling_thread.start()
        logger.info("Profiling thread started")

    def stop(self) -> None:
        """주기적 프로파일링 중지"""
        if not self.config.enabled or self._profiling_thread is None:
            return

        self._stop_event.set()
        self._profiling_thread.join(timeout=5.0)
        if self._profiling_thread.is_alive():
            logger.warning("Profiling thread did not terminate properly")
        else:
            logger.info("Profiling thread stopped")

        # 최종 프로파일 결과 저장
        self._save_profile_results()

    def _profiling_loop(self) -> None:
        """주기적 프로파일링 루프"""
        last_profile_time = time.time()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # 지정된 간격마다 프로파일링 수행
                if current_time - last_profile_time >= self.config.profile_interval:
                    self._perform_profiling()
                    last_profile_time = current_time

                # 잠시 대기
                time.sleep(1.0)

            except Exception as err:
                logger.error(f"Error in profiling loop: {str(err)}", exc_info=True)
                time.sleep(5.0)  # 오류 발생 시 더 오래 대기

    def _perform_profiling(self) -> None:
        """프로파일링 작업 수행"""
        logger.debug("Performing system profiling")

        # 메모리 스냅샷 생성
        if self.config.memory_profiling:
            try:
                memory_snapshot = MemorySnapshot.capture()
                self.memory_snapshots.append(memory_snapshot)

                # 최대 100개만 유지
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots.pop(0)

                if self.config.verbose:
                    logger.info(f"Memory snapshot: Process RAM: {memory_snapshot.process_ram_used_gb:.2f} GB, "
                                f"System RAM: {memory_snapshot.system_ram_percent:.1f}%")
            except Exception as err:
                logger.error(f"Error capturing memory snapshot: {str(err)}", exc_info=True)

        # PyTorch 프로파일링
        if self.config.torch_profiling and HAS_TORCH_PROFILER and torch.cuda.is_available():
            try:
                self._profile_torch_operations()
            except Exception as err:
                logger.error(f"Error in PyTorch profiling: {str(err)}", exc_info=True)

        # 프로파일 결과 저장
        if self.config.log_to_file:
            self._save_profile_results()

    def _profile_torch_operations(self) -> None:
        """PyTorch 연산 프로파일링"""
        if not HAS_TORCH_PROFILER:
            return

        logger.debug("Starting PyTorch operations profiling")

        # 이전 세션 종료
        if self.torch_profiler_session is not None:
            try:
                self.torch_profiler_session.__exit__(None, None, None)
            except Exception as err:
                logger.warning(f"Torch profiler session 종료 중 오류 발생: {err}")
                # 프로파일러 세션 종료는 실패해도 프로그램 동작에 영향을 주지 않으므로 무시
                pass

        # 새 프로파일링 세션 시작
        try:
            profile_path = os.path.join(self.config.profile_dir, f"torch_profile_{int(time.time())}")

            self.torch_profiler_session = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA
                ] if torch.cuda.is_available() else [torch_profiler.ProfilerActivity.CPU],
                schedule=torch_profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2
                ),
                on_trace_ready=torch_profiler.tensorboard_trace_handler(profile_path),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )

            self.torch_profiler_session.__enter__()
            logger.debug(f"PyTorch profiling session started, results will be saved to {profile_path}")

        except Exception as err:
            logger.error(f"Failed to start PyTorch profiling: {str(err)}", exc_info=True)
            self.torch_profiler_session = None

    def _save_profile_results(self) -> None:
        """프로파일 결과 저장"""
        if not self.config.log_to_file:
            return

        # 현재 시간 기반 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 메모리 스냅샷 저장
        if self.memory_snapshots:
            try:
                memory_file = os.path.join(self.config.profile_dir, f"memory_profile_{timestamp}.json")
                with open(memory_file, 'w', encoding='utf-8') as func_f:
                    json.dump(
                        [asdict(snapshot) for snapshot in self.memory_snapshots[-10:]],  # 최근 10개만 저장
                        func_f,
                        indent=2
                    )
                logger.debug(f"Memory snapshots saved to {memory_file}")
            except Exception as errr:
                logger.error(f"Failed to save memory snapshots: {str(errr)}", exc_info=True)

        # 함수 프로파일 저장
        if self.function_profiles:
            try:
                functions_file = os.path.join(self.config.profile_dir, f"function_profile_{timestamp}.json")
                with open(functions_file, 'w', encoding='utf-8') as func_f:
                    json.dump(
                        {k: asdict(v) for k, v in self.function_profiles.items()},
                        func_f,
                        indent=2
                    )
                logger.debug(f"Function profiles saved to {functions_file}")
            except Exception as err:
                logger.error(f"Failed to save function profiles: {str(err)}", exc_info=True)

    def profile_function(self, p_func: Callable) -> Callable:
        """
        함수 실행 시간 프로파일링 데코레이터

        Args:
            p_func: 프로파일링할 함수

        Returns:
            프로파일링이 적용된 함수
        """
        if not self.config.enabled:
            return p_func

        @functools.wraps(p_func)
        def wrapper(*args, **kwargs):
            if not self.config.enabled:
                return p_func(*args, **kwargs)

            # 함수 정보
            func_name = p_func.__name__
            # module_name = p_func.__module__
            # p_func를 Callable 대신 Any로 캐스팅
            module_name = cast(Any, p_func).__module__
            key = f"{module_name}.{func_name}"

            # 함수 프로파일 생성 또는 가져오기
            if key not in self.function_profiles:
                self.function_profiles[key] = FunctionProfile(
                    function_name=func_name,
                    module_name=module_name
                )

            # 실행 시간 측정
            start_time = time.time()
            try:
                result = p_func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time

                # 프로파일 업데이트
                self.function_profiles[key].update(execution_time)

                if self.config.verbose:
                    logger.debug(f"Function {key} executed in {execution_time:.6f}s")

        return wrapper

    async def get_memory_report(self) -> Dict[str, Any]:
        """
        현재 메모리 사용 보고서 생성

        Returns:
            메모리 사용 보고서
        """
        if not self.config.enabled:
            return {"status": "profiler_disabled"}

        try:
            # 최신 스냅샷 생성
            current_snapshot = MemorySnapshot.capture()

            # 이전 스냅샷과 비교 (있는 경우)
            prev_snapshot = self.memory_snapshots[-1] if self.memory_snapshots else None

            report = {
                "timestamp": datetime.now().isoformat(),
                "current": asdict(current_snapshot),
                "history": {
                    "snapshots_count": len(self.memory_snapshots),
                    "time_span": f"{len(self.memory_snapshots) * self.config.profile_interval} "
                                 f"seconds" if self.memory_snapshots else "0 seconds"
                }
            }

            # 변화량 계산 (이전 스냅샷이 있는 경우)
            if prev_snapshot:
                ram_change = current_snapshot.process_ram_used_gb - prev_snapshot.process_ram_used_gb
                report["changes"] = {
                    "process_ram_gb": f"{ram_change:.3f} "
                                      f"({'+' if ram_change >= 0 else ''}"
                                      f"{ram_change / prev_snapshot.process_ram_used_gb * 100:.1f}%"
                                      f")" if prev_snapshot.process_ram_used_gb > 0 else "N/A"
                }

                # GPU 메모리 변화 (장치가 동일한 경우)
                if current_snapshot.gpu_memory and prev_snapshot.gpu_memory:
                    gpu_changes = []
                    for idx, (curr, prev) in enumerate(zip(current_snapshot.gpu_memory, prev_snapshot.gpu_memory)):
                        if curr["id"] == prev["id"]:
                            mem_change = curr["memory_allocated_gb"] - prev["memory_allocated_gb"]
                            gpu_changes.append({
                                "id": curr["id"],
                                "memory_change_gb": f"{mem_change:.3f}",
                                "change_percent": f"{'+' if mem_change >= 0 else ''}"
                                                  f"{mem_change / prev['memory_allocated_gb'] * 100:.1f}%" if
                                prev['memory_allocated_gb'] > 0 else "N/A"
                            })

                    if gpu_changes:
                        report["changes"]["gpu_memory"] = gpu_changes

            return report

        except Exception as err:
            logger.error(f"Error generating memory report: {str(err)}", exc_info=True)
            return {
                "status": "error",
                "error": str(err),
                "timestamp": datetime.now().isoformat()
            }

    async def get_function_stats(self, top_n: int = 20) -> Dict[str, Any]:
        """
        함수 프로파일링 통계 보고서 생성

        Args:
            top_n: 반환할 상위 함수 수

        Returns:
            함수 프로파일링 통계
        """
        if not self.config.enabled:
            return {"status": "profiler_disabled"}

        try:
            # 총 실행 시간 기준 정렬
            sorted_profiles = sorted(
                self.function_profiles.values(),
                key=lambda x: x.total_time,
                reverse=True
            )

            # 상위 N개 함수만 선택
            top_functions = [asdict(f_profile) for f_profile in sorted_profiles[:top_n]]

            # 요약 통계
            total_functions = len(self.function_profiles)
            total_calls = sum(f_profile.call_count for f_profile in self.function_profiles.values())
            total_time = sum(f_profile.total_time for f_profile in self.function_profiles.values())

            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_functions": total_functions,
                    "total_calls": total_calls,
                    "total_time": total_time
                },
                "top_functions": top_functions
            }

        except Exception as err:
            logger.error(f"Error generating function stats: {str(err)}", exc_info=True)
            return {
                "status": "error",
                "error": str(err),
                "timestamp": datetime.now().isoformat()
            }

    @classmethod
    async def get_system_stats(cls) -> Dict[str, Any]:
        """
        시스템 상태 통계 보고서 생성

        Returns:
            시스템 상태 보고서
        """
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=0.1),
                    "count_logical": psutil.cpu_count(),
                    "count_physical": psutil.cpu_count(logical=False),
                    "load_avg": psutil.getloadavg()
                },
                "memory": {
                    "total_gb": psutil.virtual_memory().total / (1024 ** 3),
                    "available_gb": psutil.virtual_memory().available / (1024 ** 3),
                    "used_gb": psutil.virtual_memory().used / (1024 ** 3),
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total_gb": psutil.disk_usage('/').total / (1024 ** 3),
                    "used_gb": psutil.disk_usage('/').used / (1024 ** 3),
                    "free_gb": psutil.disk_usage('/').free / (1024 ** 3),
                    "percent": psutil.disk_usage('/').percent
                },
                "process": {
                    "pid": os.getpid(),
                    "memory_gb": psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3),
                    "cpu_percent": psutil.Process(os.getpid()).cpu_percent(interval=0.1),
                    "threads": len(psutil.Process(os.getpid()).threads())
                }
            }

            # GPU 정보 추가
            if torch.cuda.is_available():
                gpu_stats = []
                for idx in range(torch.cuda.device_count()):
                    gpu_info = {
                        "id": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "memory_allocated_gb": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3)
                    }
                    gpu_stats.append(gpu_info)

                stats["gpu"] = gpu_stats

            return stats

        except Exception as err:
            logger.error(f"Error generating system stats: {str(err)}", exc_info=True)
            return {
                "status": "error",
                "error": str(err),
                "timestamp": datetime.now().isoformat()
            }


# 활성 프로파일러 인스턴스 (싱글톤)
_active_profiler = None


def get_profiler() -> SystemProfiler:
    """
    시스템 프로파일러 인스턴스 가져오기 (싱글톤)

    Returns:
        SystemProfiler 인스턴스
    """
    global _active_profiler

    if _active_profiler is None:
        # 설정에서 프로파일링 설정 가져오기
        local_profiler_config = ProfilerConfig(
            enabled=config.monitoring.enabled,
            profile_interval=config.monitoring.profile_interval,
            memory_profiling=config.monitoring.record_memory,
            torch_profiling=torch.cuda.is_available(),
            log_to_file=True,
            profile_dir="./profiles",
            verbose=config.logging.level.upper() == "DEBUG"
        )

        _active_profiler = SystemProfiler(local_profiler_config)
        _active_profiler.start()

    return _active_profiler


def profile(func_):
    """
    함수 프로파일링 데코레이터

    Args:
        func_: 프로파일링할 함수

    Returns:
        프로파일링이 적용된 함수
    """
    return get_profiler().profile_function(func_)


# 컨텍스트 관리자를 통한 코드 블록 프로파일링
@contextlib.contextmanager
def profile_block(name: str):
    """
    코드 블록 프로파일링을 위한 컨텍스트 관리자

    Args:
        name: 프로파일링 블록 이름
    """
    profiler_ = get_profiler()
    if not profiler_.config.enabled:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time

        # 함수 프로파일로 기록
        key = f"block.{name}"
        if key not in profiler_.function_profiles:
            profiler_.function_profiles[key] = FunctionProfile(
                function_name=name,
                module_name="block"
            )

        profiler_.function_profiles[key].update(execution_time)

        if profiler_.config.verbose:
            logger.debug(f"Block {name} executed in {execution_time:.6f}s")


# API 엔드포인트용 함수
async def get_profiler_memory_report() -> Dict[str, Any]:
    """
    API를 통한 메모리 보고서 가져오기

    Returns:
        현재 메모리 사용 보고서
    """
    profiler_ = get_profiler()
    return await profiler_.get_memory_report()


async def get_profiler_function_stats() -> Dict[str, Any]:
    """
    API를 통한 함수 통계 가져오기

    Returns:
        함수 실행 시간 통계
    """
    profiler_ = get_profiler()
    return await profiler_.get_function_stats()


async def get_profiler_system_stats() -> Dict[str, Any]:
    """
    API를 통한 시스템 상태 통계 가져오기

    Returns:
        시스템 상태 보고서
    """
    profiler_ = get_profiler()
    return await profiler_.get_system_stats()


# 프로파일러 초기화
def init_profiling():
    """
    프로파일링 시스템 초기화

    서버 시작 시 호출됨

    Returns:
        활성화된 프로파일러 인스턴스 또는 None
    """
    if config.monitoring.enabled:
        try:
            profiler_ = get_profiler()
            logger.info("Profiling system initialized")
            return profiler_
        except Exception as err:
            logger.error(f"Failed to initialize profiling: {str(err)}", exc_info=True)
            return None
    else:
        logger.info("Profiling is disabled in configuration")
        return None


# 모듈 테스트 코드
if __name__ == "__main__":
    # 로깅 레벨 설정
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("profiler-test")
    logger.info("프로파일러 테스트 시작")

    # 결과 저장 디렉토리 생성
    import os
    from datetime import datetime

    output_dir = f"./profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"결과 저장 디렉토리: {output_dir}")

    # 프로파일러 설정 및 초기화
    profiler_config = ProfilerConfig(
        enabled=True,
        profile_interval=5,
        memory_profiling=True,
        torch_profiling=torch.cuda.is_available(),
        log_to_file=True,
        profile_dir=output_dir,
        verbose=True
    )

    profiler = SystemProfiler(profiler_config)
    profiler.start()
    logger.info("프로파일러 시작됨")

    try:
        # ===== 1. CPU 부하 테스트 함수 =====
        @profile
        def cpu_intensive_test(num):
            """CPU 집약적인 작업 테스트"""
            logger.info(f"CPU 부하 테스트 시작 (n={num})")
            result = 0
            for idx in range(num):
                result += idx ** 2
            return result


        # ===== 2. 메모리 사용 테스트 함수 =====
        @profile
        def memory_test(size_mb):
            """메모리 사용량 테스트"""
            logger.info(f"메모리 테스트 시작 (size={size_mb}MB)")
            # size_mb 크기의 메모리 할당
            data = [0] * (size_mb * 131072)  # 약 1MB = 1024 * 1024 / 8 (bytes per int)
            return sum(data[:100])  # 계산을 통해 최적화 방지


        # ===== 3. 의도적인 메모리 누수 시뮬레이션 =====
        memory_leak_storage = []


        @profile
        def simulate_memory_leak(size_mb, rounds=5):
            """의도적인 메모리 누수 시뮬레이션"""
            logger.info(f"메모리 누수 시뮬레이션 시작 (size={size_mb}MB, rounds={rounds})")
            for idx in range(rounds):
                # 전역 변수에 저장하여 참조가 유지되도록 함
                data = "X" * (size_mb * 1024 * 1024)  # size_mb MB
                memory_leak_storage.append(data)
                logger.info(f"  누수 라운드 {idx + 1}/{rounds}: 현재 {len(memory_leak_storage)}개 객체")
                time.sleep(1)  # 메모리 사용량 변화 관찰을 위한 지연


        # ===== 4. GPU 테스트 함수 =====
        def gpu_test():
            """GPU 사용 테스트 (CUDA 가능한 경우)"""
            if not torch.cuda.is_available():
                logger.warning("CUDA를 사용할 수 없어 GPU 테스트를 건너뜁니다")
                return

            logger.info(f"GPU 테스트 시작 (장치: {torch.cuda.get_device_name(0)})")

            @profile
            def gpu_matrix_multiply(size_gpu):
                """GPU 행렬 곱셈 테스트"""
                logger.info(f"GPU 행렬 곱셈 테스트 (크기: {size_gpu}x{size_gpu})")
                x = torch.rand(size_gpu, size_gpu, device="cuda")
                y = torch.rand(size_gpu, size_gpu, device="cuda")

                with profile_block("gpu_matrix_multiply"):
                    z = torch.matmul(x, y)

                return z.sum().item()

            @profile
            def gpu_memory_test(size_gpu_mb):
                """GPU 메모리 할당 테스트"""
                logger.info(f"GPU 메모리 할당 테스트 (크기: {size_gpu_mb}MB)")
                # float32 텐서 할당 (4bytes per float)
                num_elements = int(size_gpu_mb * 1024 * 1024 / 4)
                size_ = int(num_elements ** 0.5)

                with profile_block("gpu_memory_allocation"):
                    x = torch.rand(size_, size_, device="cuda")

                with profile_block("gpu_operations"):
                    y = torch.sin(x) + torch.cos(x)

                return y.sum().item()

            # 다양한 크기로 GPU 테스트 실행
            for size_t in [1000, 2000, 3000]:
                gpu_matrix_multiply(size_t)
                time.sleep(1)  # GPU 메모리 해제를 위한 지연

            # GPU 메모리 테스트
            for size_mb in [100, 200, 400]:
                gpu_memory_test(size_mb)
                time.sleep(1)


        def get_sync_system_stats():
            """동기 방식으로 시스템 상태를 가져오는 래퍼 함수"""
            return asyncio.run(profiler.get_system_stats())


        def get_sync_memory_report():
            """동기 방식으로 시스템 상태를 가져오는 래퍼 함수"""
            return asyncio.run(profiler.get_memory_report())


        def get_sync_function_stats():
            """동기 방식으로 시스템 상태를 가져오는 래퍼 함수"""
            return asyncio.run(profiler.get_function_stats(top_n=5))


        # ===== 5. 결과 시각화 함수 =====
        def visualize_results():
            """프로파일링 결과 시각화"""
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                logger.info("결과 시각화 시작")

                # 메모리 스냅샷 시각화
                if len(profiler.memory_snapshots) > 1:
                    # 시간에 따른 프로세스 메모리 사용량
                    plt.figure(figsize=(10, 6))
                    timestamps = list(range(len(profiler.memory_snapshots)))
                    memory_values = [snap.process_ram_used_gb for snap in profiler.memory_snapshots]

                    plt.plot(timestamps, memory_values, 'b-', marker='o')
                    plt.title('프로세스 메모리 사용량 변화')
                    plt.xlabel('스냅샷 순서')
                    plt.ylabel('메모리 사용량 (GB)')
                    plt.grid(True)
                    plt.savefig(f"{output_dir}/memory_usage.png")
                    logger.info(f"메모리 사용량 그래프 저장됨: {output_dir}/memory_usage.png")

                    # GPU 메모리 시각화 (CUDA 가능한 경우)
                    if torch.cuda.is_available() and profiler.memory_snapshots[0].gpu_memory:
                        plt.figure(figsize=(10, 6))
                        gpu_memory_values = []

                        # 각 GPU별 메모리 사용량 추출
                        for gpu_idx in range(torch.cuda.device_count()):
                            values = []
                            for snap in profiler.memory_snapshots:
                                if gpu_idx < len(snap.gpu_memory):
                                    values.append(snap.gpu_memory[gpu_idx].get('memory_allocated_gb', 0))
                                else:
                                    values.append(0)
                            gpu_memory_values.append(values)

                        # 각 GPU별 그래프 그리기
                        colors = ['b', 'r', 'g', 'c', 'm', 'y']
                        for idx, values in enumerate(gpu_memory_values):
                            plt.plot(timestamps, values,
                                     color=colors[idx % len(colors)],
                                     marker='o',
                                     label=f'GPU {idx}')

                        plt.title('GPU 메모리 사용량 변화')
                        plt.xlabel('스냅샷 순서')
                        plt.ylabel('할당된 메모리 (GB)')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f"{output_dir}/gpu_memory_usage.png")
                        logger.info(f"GPU 메모리 그래프 저장됨: {output_dir}/gpu_memory_usage.png")

                # 함수 실행 시간 시각화
                if profiler.function_profiles:
                    plt.figure(figsize=(12, 8))

                    # 상위 10개 함수만 선택
                    top_functions = sorted(
                        profiler.function_profiles.values(),
                        key=lambda x: x.total_time,
                        reverse=True
                    )[:10]

                    function_names = [f"{t_func.module_name}.{t_func.function_name}" for t_func in top_functions]
                    avg_times = [t_func.avg_time for t_func in top_functions]
                    total_times = [t_func.total_time for t_func in top_functions]

                    # 평균 실행 시간 그래프
                    plt.subplot(2, 1, 1)
                    plt.barh(range(len(function_names)), avg_times, color='skyblue')
                    plt.yticks(range(len(function_names)), function_names)
                    plt.title('함수별 평균 실행 시간')
                    plt.xlabel('시간 (초)')
                    plt.grid(axis='x')

                    # 총 실행 시간 그래프
                    plt.subplot(2, 1, 2)
                    plt.barh(range(len(function_names)), total_times, color='salmon')
                    plt.yticks(range(len(function_names)), function_names)
                    plt.title('함수별 총 실행 시간')
                    plt.xlabel('시간 (초)')
                    plt.grid(axis='x')

                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/function_times.png")
                    logger.info(f"함수 실행 시간 그래프 저장됨: {output_dir}/function_times.png")

            except ImportError:
                logger.warning("matplotlib이 설치되지 않아 시각화를 건너뜁니다")
            except Exception as err:
                logger.error(f"시각화 중 오류 발생: {err}", exc_info=True)


        # ===== 6. 테스트 실행 =====
        logger.info("===== 테스트 시작 =====")

        # CPU 부하 테스트
        logger.info("CPU 부하 테스트 실행")
        for n in [1000000, 5000000, 10000000]:
            with profile_block(f"cpu_test_{n}"):
                cpu_intensive_test(n)

        # 메모리 사용 테스트
        logger.info("메모리 사용 테스트 실행")
        for size in [50, 100, 200]:
            with profile_block(f"memory_test_{size}MB"):
                memory_test(size)

        # GPU 테스트 (CUDA 가능한 경우)
        logger.info("GPU 테스트 실행")
        gpu_test()

        # 메모리 누수 시뮬레이션
        logger.info("메모리 누수 시뮬레이션 실행")
        simulate_memory_leak(20, rounds=3)

        # 시스템 상태 모니터링
        for i in range(3):
            system_stats = get_sync_system_stats()

            logger.info(f"시스템 상태 #{i + 1}: "
                        f"CPU {system_stats['cpu']['usage_percent']:.1f}%, "
                        f"RAM {system_stats['memory']['percent']:.1f}%")
            time.sleep(2)

        # ===== 7. 결과 수집 및 보고서 생성 =====
        logger.info("===== 결과 수집 중 =====")

        # 메모리 보고서
        memory_report = get_sync_memory_report()
        logger.info(f"프로세스 메모리: {memory_report['current']['process_ram_used_gb']:.2f} GB")

        # 함수 통계
        function_stats = get_sync_function_stats()
        logger.info("함수 실행 시간 통계 (상위 5개):")
        for i, func in enumerate(function_stats.get('top_functions', [])):
            logger.info(f"  {i + 1}. {func['module_name']}.{func['function_name']}: "
                        f"{func['call_count']}회 호출, "
                        f"총 {func['total_time']:.4f}초, "
                        f"평균 {func['avg_time']:.6f}초")

        # 결과 JSON 저장
        result_data = {
            "memory_report": memory_report,
            "function_stats": function_stats,
            "system_stats": profiler.get_system_stats()
        }

        with open(f"{output_dir}/profile_results.json", "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"결과 JSON 저장됨: {output_dir}/profile_results.json")

        # 결과 시각화
        visualize_results()

        logger.info(f"===== 테스트 완료 =====")
        logger.info(f"모든 결과는 {output_dir} 디렉토리에 저장되었습니다")

    except KeyboardInterrupt:
        logger.info("사용자에 의한 테스트 중단")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)
    finally:
        # 프로파일러 정리
        profiler.stop()
        logger.info("프로파일러 종료됨")

        # 중요 결과 출력
        print("\n===== 프로파일링 요약 =====")
        print(f"결과 디렉토리: {output_dir}")

        if profiler.function_profiles:
            # 가장 오래 걸린 함수 찾기
            slowest_func = max(profiler.function_profiles.values(), key=lambda x: x.total_time)
            print(f"가장 오래 걸린 함수: {slowest_func.module_name}.{slowest_func.function_name}")
            print(f"  - 총 시간: {slowest_func.total_time:.4f}초")
            print(f"  - 평균 시간: {slowest_func.avg_time:.6f}초")
            print(f"  - 호출 횟수: {slowest_func.call_count}회")

        if profiler.memory_snapshots:
            initial = profiler.memory_snapshots[0].process_ram_used_gb
            final = profiler.memory_snapshots[-1].process_ram_used_gb
            change = final - initial
            print(f"메모리 사용량 변화: {initial:.2f} GB → {final:.2f} GB ({change:.2f} GB)")

        if torch.cuda.is_available():
            print("\nGPU 메모리 현황:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"  GPU {i}: 할당됨 {allocated:.2f} GB, 예약됨 {reserved:.2f} GB")
