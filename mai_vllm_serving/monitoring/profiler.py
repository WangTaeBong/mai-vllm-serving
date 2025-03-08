"""
리팩토링된 mai-vllm-serving 성능 프로파일링 모듈
시스템 성능 측정 및 분석을 위한 상세 프로파일링 기능 제공
"""
import asyncio
import contextlib
import functools
import gc
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, cast

import psutil
import py3nvml.py3nvml as nvml
import torch
import torch.profiler as torch_profiler

# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config
# logging_utils의 고급 로깅 기능 임포트
from mai_vllm_serving.utils.logging_utils import (
    get_logger,
    TimingContext,
    with_logging_context
)

# 의존성 관리 개선: 버전 명시 및 그룹화
REQUIRED_PACKAGES = {
    "torch": {"min_version": "1.10.0", "recommended": "2.0.0"},
    "psutil": {"min_version": "5.8.0", "recommended": "5.9.0"}
}

# 선택적 의존성 관리 개선
OPTIONAL_PACKAGES = {
    "torch_profiler": {"module": "torch.profiler", "version": None},
    "gputil": {"module": "GPUtil", "version": "1.4.0"},
    "memory_profiler": {"module": "memory_profiler", "version": "0.60.0"},
    "nvml": {"module": "py3nvml.py3nvml", "version": "0.2.7"}
}

# 의존성 가져오기 및 상태 추적
PACKAGE_STATUS = {
    "torch_profiler": True,
    "gputil": True,
    "memory_profiler": True,
    "nvml": True
}

# 설정 객체 가져오기
config = get_config()

# 구조화된 로거 가져오기 (로깅 유틸리티 사용)
logger = get_logger("profiler")

# 세부 로깅을 위한 서브 로거들
memory_logger = get_logger("profiler.memory")
function_logger = get_logger("profiler.function")
gpu_logger = get_logger("profiler.gpu")


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
class GPUInfo:
    """GPU 정보"""
    id: int
    name: str
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    memory_cached_gb: float = 0.0
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature: float = 0.0

    @classmethod
    def collect_gpu_info(cls, idx: int) -> 'GPUInfo':
        """특정 GPU의 정보 수집"""
        try:
            gpu_info = cls(
                id=idx,
                name=torch.cuda.get_device_name(idx),
                memory_allocated_gb=torch.cuda.memory_allocated(idx) / (1024 ** 3),
                memory_reserved_gb=torch.cuda.memory_reserved(idx) / (1024 ** 3),
                memory_cached_gb=torch.cuda.memory_reserved(idx) / (1024 ** 3) if hasattr(torch.cuda,
                                                                                          'memory_reserved') else 0
            )

            # NVML 정보 추가 (가능한 경우)
            if PACKAGE_STATUS["nvml"]:
                try:
                    with TimingContext(gpu_logger.debug, f"NVML GPU {idx} info collection"):
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
                        info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info.total_memory_gb = info.total / (1024 ** 3)
                        gpu_info.used_memory_gb = info.used / (1024 ** 3)
                        gpu_info.free_memory_gb = info.free / (1024 ** 3)

                        # GPU 사용률
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_info.gpu_utilization = util.gpu
                        gpu_info.memory_utilization = util.memory

                        # 온도
                        gpu_info.temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        nvml.nvmlShutdown()
                except Exception as err:
                    gpu_logger.debug(f"NVML 정보 수집 중 오류 발생: {str(err)}",
                                     context={"error_type": type(err).__name__, "gpu_idx": idx})

            return gpu_info
        except Exception as err:
            gpu_logger.error(f"GPU {idx} 정보 수집 중 오류 발생: {str(err)}",
                             context={"error_type": type(err).__name__, "gpu_idx": idx})
            # 기본 정보라도 반환
            return cls(id=idx, name=f"Unknown-GPU-{idx}")


@dataclass
class MemorySnapshot:
    """메모리 사용 스냅샷"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_ram_used_gb: float = 0.0
    system_ram_total_gb: float = 0.0
    system_ram_percent: float = 0.0
    process_ram_used_gb: float = 0.0
    python_objects: Dict[str, int] = field(default_factory=dict)
    gpu_memory: List[GPUInfo] = field(default_factory=list)

    @staticmethod
    def capture() -> 'MemorySnapshot':
        """
        현재 메모리 사용 상태 캡처

        Returns:
            메모리 스냅샷 객체
        """
        with TimingContext(memory_logger.debug, "메모리 스냅샷 캡처"):
            snapshot = MemorySnapshot()

            # 시스템 RAM 정보 수집
            snapshot._collect_system_ram_info()

            # 프로세스 메모리 정보 수집
            snapshot._collect_process_info()

            # Python 객체 정보 수집 (memory_profiler 필요)
            if PACKAGE_STATUS["memory_profiler"]:
                with TimingContext(memory_logger.debug, "Python 객체 정보 수집"):
                    snapshot._collect_python_objects()

            # GPU 메모리 정보 수집
            if torch.cuda.is_available():
                with TimingContext(memory_logger.debug, "GPU 메모리 정보 수집"):
                    snapshot._collect_gpu_info()

            memory_logger.debug("메모리 스냅샷 캡처 완료",
                                context={
                                    "system_ram_percent": f"{snapshot.system_ram_percent:.1f}%",
                                    "process_ram_gb": f"{snapshot.process_ram_used_gb:.2f}GB"
                                })
            return snapshot

    def _collect_system_ram_info(self) -> None:
        """시스템 RAM 정보 수집"""
        mem = psutil.virtual_memory()
        self.system_ram_used_gb = mem.used / (1024 ** 3)
        self.system_ram_total_gb = mem.total / (1024 ** 3)
        self.system_ram_percent = mem.percent

    def _collect_process_info(self) -> None:
        """현재 프로세스 메모리 정보 수집"""
        pid = os.getpid()
        process = psutil.Process(pid)
        self.process_ram_used_gb = process.memory_info().rss / (1024 ** 3)

    def _collect_python_objects(self) -> None:
        """Python 객체 카운트 수집"""
        objects = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            if obj_type in objects:
                objects[obj_type] += 1
            else:
                objects[obj_type] = 1

        # 가장 많은 객체 타입 상위 20개만 저장
        self.python_objects = dict(
            sorted(objects.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        # 객체 분포 로깅 (디버그 수준)
        if memory_logger.level <= 10:  # DEBUG=10
            top_obj_info = ", ".join([f"{k}:{v}" for k, v in list(self.python_objects.items())[:5]])
            memory_logger.debug(f"상위 객체 분포: {top_obj_info}")

    def _collect_gpu_info(self) -> None:
        """GPU 메모리 정보 수집"""
        for idx in range(torch.cuda.device_count()):
            gpu_info = GPUInfo.collect_gpu_info(idx)
            self.gpu_memory.append(gpu_info)

            # GPU 정보 로깅 (INFO 레벨)
            gpu_logger.info(f"GPU {idx} 메모리 상태",
                            context={
                                "gpu_id": idx,
                                "name": gpu_info.name,
                                "memory_used_gb": f"{gpu_info.memory_allocated_gb:.2f}GB",
                                "utilization": f"{gpu_info.gpu_utilization:.1f}%",
                                "temperature": f"{gpu_info.temperature:.1f}°C"
                            })


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
            logger.info("프로파일러가 비활성화되었습니다")
            return

        # 함수 프로파일 저장소
        self.function_profiles: Dict[str, FunctionProfile] = {}

        # 메모리 스냅샷 이력
        self.memory_snapshots: List[MemorySnapshot] = []

        # 프로파일 결과 저장 디렉토리 생성
        if self.config.log_to_file:
            os.makedirs(self.config.profile_dir, exist_ok=True)
            logger.info(f"프로파일 결과 디렉토리 생성: {self.config.profile_dir}")

        # 주기적 프로파일링 스레드
        self._stop_event = threading.Event()
        self._profiling_thread = None

        # PyTorch 프로파일링 세션
        self.torch_profiler_session = None

        # 로그 레벨에 따른 상세 로깅 설정
        if self.config.verbose or get_config().logging.level.upper() == "DEBUG":  # DEBUG=10
            logger.debug("상세 로깅이 활성화되었습니다")

        logger.info(f"시스템 프로파일러가 초기화되었습니다 (간격: {self.config.profile_interval}초)",
                    context={
                        "profile_interval": self.config.profile_interval,
                        "memory_profiling": self.config.memory_profiling,
                        "torch_profiling": self.config.torch_profiling
                    })

    def start(self) -> None:
        """주기적 프로파일링 시작"""
        if not self.config.enabled:
            return

        if self._profiling_thread is not None and self._profiling_thread.is_alive():
            logger.warning("프로파일링 스레드가 이미 실행 중입니다")
            return

        self._stop_event.clear()
        self._profiling_thread = threading.Thread(
            target=self._profiling_loop,
            daemon=True,
            name="ProfilingThread"
        )
        self._profiling_thread.start()
        logger.info("프로파일링 스레드가 시작되었습니다")

    def stop(self) -> None:
        """주기적 프로파일링 중지"""
        if not self.config.enabled or self._profiling_thread is None:
            return

        with TimingContext(logger.debug, "프로파일링 정리 작업"):
            self._stop_event.set()
            self._profiling_thread.join(timeout=5.0)
            if self._profiling_thread.is_alive():
                logger.warning("프로파일링 스레드가 제대로 종료되지 않았습니다")
            else:
                logger.info("프로파일링 스레드가 중지되었습니다")

            # 최종 프로파일 결과 저장
            self._save_profile_results()

    def _profiling_loop(self) -> None:
        """주기적 프로파일링 루프"""
        last_profile_time = time.time()
        loop_count = 0

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                loop_count += 1

                # 지정된 간격마다 프로파일링 수행
                if current_time - last_profile_time >= self.config.profile_interval:
                    with TimingContext(logger.debug, f"프로파일링 실행 #{loop_count}"):
                        self._perform_profiling()
                    last_profile_time = current_time

                # 잠시 대기 (10% 간격으로 중지 이벤트 체크)
                check_interval = min(1.0, self.config.profile_interval / 10)
                time.sleep(check_interval)

            except Exception as err:
                logger.error(f"프로파일링 루프 오류: {str(err)}",
                             context={"error_type": type(err).__name__},
                             exc_info=True)
                time.sleep(5.0)  # 오류 발생 시 더 오래 대기

    def _perform_profiling(self) -> None:
        """프로파일링 작업 수행"""
        logger.debug("시스템 프로파일링 수행 중")

        # 메모리 스냅샷 생성
        if self.config.memory_profiling:
            with TimingContext(memory_logger.debug, "메모리 스냅샷 캡처"):
                self._capture_memory_snapshot()

        # PyTorch 프로파일링
        if self.config.torch_profiling and PACKAGE_STATUS["torch_profiler"] and torch.cuda.is_available():
            with TimingContext(gpu_logger.debug, "PyTorch 연산 프로파일링"):
                self._profile_torch_operations()

        # 함수 프로파일 통계 로깅
        if self.function_profiles:
            top_functions = sorted(
                self.function_profiles.values(),
                key=lambda x: x.total_time,
                reverse=True
            )[:5]  # 상위 5개만

            function_logger.info("함수 프로파일 요약",
                                 context={
                                     "total_functions": len(self.function_profiles),
                                     "top_functions": [
                                         {
                                             "name": f"{func.module_name}.{func.function_name}",
                                             "calls": func.call_count,
                                             "total_time": f"{func.total_time:.3f}s",
                                             "avg_time": f"{func.avg_time:.6f}s"
                                         } for func in top_functions
                                     ]
                                 })

        # 프로파일 결과 저장
        if self.config.log_to_file:
            with TimingContext(logger.debug, "프로파일 결과 저장"):
                self._save_profile_results()

    def _capture_memory_snapshot(self) -> None:
        """메모리 스냅샷 캡처"""
        try:
            memory_snapshot = MemorySnapshot.capture()
            self.memory_snapshots.append(memory_snapshot)

            # 최대 100개만 유지
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots.pop(0)

            # 상세 로깅
            if self.config.verbose or memory_logger.level <= 20:  # INFO=20
                memory_logger.info(f"메모리 스냅샷: 프로세스 RAM: {memory_snapshot.process_ram_used_gb:.2f} GB, "
                                   f"시스템 RAM: {memory_snapshot.system_ram_percent:.1f}%")
        except Exception as err:
            memory_logger.error(f"메모리 스냅샷 캡처 중 오류 발생: {str(err)}",
                                context={"error_type": type(err).__name__},
                                exc_info=True)

    def _profile_torch_operations(self) -> None:
        """PyTorch 연산 프로파일링"""
        if not PACKAGE_STATUS["torch_profiler"]:
            return

        gpu_logger.debug("PyTorch 연산 프로파일링 시작")

        # 이전 세션 종료
        self._close_torch_profiler_session()

        # 새 프로파일링 세션 시작
        try:
            self._start_torch_profiler_session()
        except Exception as err:
            gpu_logger.error(f"PyTorch 프로파일링 시작 실패: {str(err)}",
                             context={"error_type": type(err).__name__},
                             exc_info=True)
            self.torch_profiler_session = None

    def _close_torch_profiler_session(self) -> None:
        """PyTorch 프로파일러 세션 종료"""
        if self.torch_profiler_session is not None:
            try:
                self.torch_profiler_session.__exit__(None, None, None)
                gpu_logger.debug("PyTorch 프로파일러 세션 종료됨")
            except Exception as err:
                gpu_logger.warning(f"PyTorch 프로파일러 세션 종료 중 오류 발생: {err}",
                                   context={"error_type": type(err).__name__})
                # 프로파일러 세션 종료는 실패해도 프로그램 동작에 영향을 주지 않으므로 무시
                pass

    def _start_torch_profiler_session(self) -> None:
        """PyTorch 프로파일러 세션 시작"""
        profile_path = os.path.join(self.config.profile_dir, f"torch_profile_{int(time.time())}")

        with TimingContext(gpu_logger.debug, "PyTorch 프로파일러 세션 초기화"):
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
            gpu_logger.debug(f"PyTorch 프로파일링 세션이 시작되었으며, 결과는 {profile_path}에 저장됩니다")

    def _save_profile_results(self) -> None:
        """프로파일 결과 저장"""
        if not self.config.log_to_file:
            return

        # 현재 시간 기반 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 메모리 스냅샷 저장
        self._save_memory_snapshots(timestamp)

        # 함수 프로파일 저장
        self._save_function_profiles(timestamp)

    def _save_memory_snapshots(self, timestamp: str) -> None:
        """메모리 스냅샷 저장"""
        if not self.memory_snapshots:
            return

        try:
            memory_file = os.path.join(self.config.profile_dir, f"memory_profile_{timestamp}.json")
            with open(memory_file, 'w', encoding='utf-8') as func_f:
                json.dump(
                    [asdict(snapshot) for snapshot in self.memory_snapshots[-10:]],  # 최근 10개만 저장
                    func_f,
                    indent=2
                )
            memory_logger.debug(f"메모리 스냅샷이 {memory_file}에 저장되었습니다")
        except Exception as errr:
            memory_logger.error(f"메모리 스냅샷 저장 실패: {str(errr)}",
                                context={"error_type": type(errr).__name__, "file": memory_file},
                                exc_info=True)

    def _save_function_profiles(self, timestamp: str) -> None:
        """함수 프로파일 저장"""
        if not self.function_profiles:
            return

        try:
            functions_file = os.path.join(self.config.profile_dir, f"function_profile_{timestamp}.json")
            with open(functions_file, 'w', encoding='utf-8') as func_f:
                json.dump(
                    {k: asdict(v) for k, v in self.function_profiles.items()},
                    func_f,
                    indent=2
                )
            function_logger.debug(f"함수 프로파일이 {functions_file}에 저장되었습니다")
        except Exception as err:
            function_logger.error(f"함수 프로파일 저장 실패: {str(err)}",
                                  context={"error_type": type(err).__name__, "file": functions_file},
                                  exc_info=True)

    @with_logging_context(module="profiler")
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
            # p_func를 Callable 대신 Any로 캐스팅
            module_name = cast(Any, p_func).__module__
            key = f"{module_name}.{func_name}"

            # 함수 프로파일 생성 또는 가져오기
            if key not in self.function_profiles:
                self.function_profiles[key] = FunctionProfile(
                    function_name=func_name,
                    module_name=module_name
                )

            # 실행 시간 측정 (TimingContext 사용)
            with TimingContext(None, f"Function {key}") as timing:
                try:
                    result = p_func(*args, **kwargs)
                    return result
                finally:
                    # 프로파일 업데이트
                    self.function_profiles[key].update(timing.duration)

                    if self.config.verbose or function_logger.level <= 10:  # DEBUG=10
                        function_logger.debug(f"함수 {key} 실행 완료",
                                              context={
                                                  "duration": f"{timing.duration:.6f}s",
                                                  "call_count": self.function_profiles[key].call_count
                                              })

        return wrapper

    async def get_memory_report(self) -> Dict[str, Any]:
        """
        현재 메모리 사용 보고서 생성

        Returns:
            메모리 사용 보고서
        """
        if not self.config.enabled:
            return {"status": "profiler_disabled"}

        with TimingContext(memory_logger.debug, "메모리 보고서 생성"):
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
                                     f"초" if self.memory_snapshots else "0초"
                    }
                }

                # 변화량 계산 (이전 스냅샷이 있는 경우)
                if prev_snapshot:
                    report["changes"] = self._calculate_memory_changes(current_snapshot, prev_snapshot)

                memory_logger.info("메모리 보고서 생성 완료",
                                   context={
                                       "ram_usage": f"{current_snapshot.process_ram_used_gb:.2f}GB",
                                       "ram_percent": f"{current_snapshot.system_ram_percent:.1f}%"
                                   })
                return report

            except Exception as err:
                memory_logger.error(f"메모리 보고서 생성 중 오류 발생: {str(err)}",
                                    context={"error_type": type(err).__name__},
                                    exc_info=True)
                return {
                    "status": "error",
                    "error": str(err),
                    "timestamp": datetime.now().isoformat()
                }

    def _calculate_memory_changes(cls, current: MemorySnapshot, previous: MemorySnapshot) -> Dict[str, Any]:
        """두 스냅샷 간의 메모리 변화량 계산"""
        changes = {}

        # RAM 변화량
        ram_change = current.process_ram_used_gb - previous.process_ram_used_gb
        changes["process_ram_gb"] = f"{ram_change:.3f} " + \
                                    f"({'+' if ram_change >= 0 else ''}" + \
                                    f"{ram_change / previous.process_ram_used_gb * 100:.1f}%" + \
                                    f")" if previous.process_ram_used_gb > 0 else "N/A"

        # GPU 메모리 변화 (장치가 동일한 경우)
        if current.gpu_memory and previous.gpu_memory:
            gpu_changes = []
            for idx, (curr_gpu, prev_gpu) in enumerate(zip(current.gpu_memory, previous.gpu_memory)):
                if curr_gpu.id == prev_gpu.id:
                    mem_change = curr_gpu.memory_allocated_gb - prev_gpu.memory_allocated_gb
                    gpu_changes.append({
                        "id": curr_gpu.id,
                        "memory_change_gb": f"{mem_change:.3f}",
                        "change_percent": f"{'+' if mem_change >= 0 else ''}"
                                          f"{mem_change / prev_gpu.memory_allocated_gb * 100:.1f}"
                                          f"%" if prev_gpu.memory_allocated_gb > 0 else "N/A"
                    })

            if gpu_changes:
                changes["gpu_memory"] = gpu_changes

                # 중요한 메모리 변화 로깅
                for change in gpu_changes:
                    if "change_percent" in change and change["change_percent"] != "N/A":
                        change_val = float(change["change_percent"].strip('%+-'))
                        if change_val > 10:  # 10% 이상 변화일 때만 로깅
                            gpu_logger.info(f"GPU {change['id']} 메모리 사용량 변화 감지",
                                            context={
                                                "gpu_id": change['id'],
                                                "change": change["change_percent"],
                                                "current_gb": f"{curr_gpu.memory_allocated_gb:.2f}GB"
                                            })

        return changes

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

        with TimingContext(function_logger.debug, "함수 통계 보고서 생성"):
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

                # 로깅
                function_logger.info("함수 통계 보고서 생성 완료",
                                     context={
                                         "total_functions": total_functions,
                                         "total_calls": total_calls,
                                         "total_time": f"{total_time:.3f}s"
                                     })

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
                function_logger.error(f"함수 통계 보고서 생성 중 오류 발생: {str(err)}",
                                      context={"error_type": type(err).__name__},
                                      exc_info=True)
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
        with TimingContext(logger.debug, "시스템 상태 통계 수집"):
            try:
                # CPU 정보 수집
                cpu_stats = cls._collect_cpu_stats()

                # 메모리 정보 수집
                memory_stats = cls._collect_memory_stats()

                # 디스크 정보 수집
                disk_stats = cls._collect_disk_stats()

                # 프로세스 정보 수집
                process_stats = cls._collect_process_stats()

                # 기본 통계 정보
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu": cpu_stats,
                    "memory": memory_stats,
                    "disk": disk_stats,
                    "process": process_stats
                }

                # GPU 정보 추가 (가능한 경우)
                if torch.cuda.is_available():
                    stats["gpu"] = cls._collect_gpu_stats()

                    # GPU 상태 로깅
                    for gpu in stats["gpu"]:
                        gpu_logger.info(f"GPU {gpu['id']} 상태",
                                        context={
                                            "id": gpu['id'],
                                            "name": gpu['name'],
                                            "memory_allocated_gb": gpu['memory_allocated_gb']
                                        })

                logger.info("시스템 상태 통계 수집 완료",
                            context={
                                "cpu_usage": f"{cpu_stats['usage_percent']:.1f}%",
                                "memory_usage": f"{memory_stats['percent']:.1f}%",
                                "process_memory": f"{process_stats['memory_gb']:.2f}GB"
                            })
                return stats

            except Exception as err:
                logger.error(f"시스템 상태 통계 수집 중 오류 발생: {str(err)}",
                             context={"error_type": type(err).__name__},
                             exc_info=True)
                return {
                    "status": "error",
                    "error": str(err),
                    "timestamp": datetime.now().isoformat()
                }

    @staticmethod
    def _collect_cpu_stats() -> Dict[str, Any]:
        """CPU 통계 수집"""
        with TimingContext(logger.debug, "CPU 통계 수집"):
            cpu_percent = psutil.cpu_percent(interval=0.1)
            count_logical = psutil.cpu_count()
            count_physical = psutil.cpu_count(logical=False)
            load_avg = psutil.getloadavg()

            logger.debug("CPU 상태 정보",
                         context={
                             "cpu_percent": f"{cpu_percent:.1f}%",
                             "cores_logical": count_logical,
                             "cores_physical": count_physical,
                             "load_avg": load_avg
                         })

            return {
                "usage_percent": cpu_percent,
                "count_logical": count_logical,
                "count_physical": count_physical,
                "load_avg": load_avg
            }

    @staticmethod
    def _collect_memory_stats() -> Dict[str, Any]:
        """메모리 통계 수집"""
        with TimingContext(memory_logger.debug, "메모리 통계 수집"):
            mem = psutil.virtual_memory()
            memory_stats = {
                "total_gb": mem.total / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "used_gb": mem.used / (1024 ** 3),
                "percent": mem.percent
            }

            memory_logger.debug("메모리 상태 정보",
                                context={
                                    "total_gb": f"{memory_stats['total_gb']:.2f}GB",
                                    "used_gb": f"{memory_stats['used_gb']:.2f}GB",
                                    "percent": f"{memory_stats['percent']:.1f}%"
                                })

            return memory_stats

    @staticmethod
    def _collect_disk_stats() -> Dict[str, Any]:
        """디스크 통계 수집"""
        with TimingContext(logger.debug, "디스크 통계 수집"):
            disk = psutil.disk_usage('/')
            disk_stats = {
                "total_gb": disk.total / (1024 ** 3),
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
                "percent": disk.percent
            }

            # 디스크 공간이 90% 이상 사용된 경우 INFO로 로깅
            if disk.percent > 90:
                logger.info("디스크 공간이 부족합니다",
                            context={
                                "used_percent": f"{disk.percent:.1f}%",
                                "free_gb": f"{disk_stats['free_gb']:.2f}GB"
                            })
            elif disk.percent > 80:
                logger.info("디스크 공간이 제한적입니다",
                            context={
                                "used_percent": f"{disk.percent:.1f}%",
                                "free_gb": f"{disk_stats['free_gb']:.2f}GB"
                            })

            return disk_stats

    @staticmethod
    def _collect_process_stats() -> Dict[str, Any]:
        """현재 프로세스 통계 수집"""
        with TimingContext(logger.debug, "프로세스 통계 수집"):
            pid = os.getpid()
            process = psutil.Process(pid)
            process_stats = {
                "pid": pid,
                "memory_gb": process.memory_info().rss / (1024 ** 3),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": len(process.threads())
            }

            logger.debug("프로세스 상태 정보",
                         context={
                             "pid": pid,
                             "memory_gb": f"{process_stats['memory_gb']:.2f}GB",
                             "threads": process_stats['threads']
                         })

            # 프로세스 메모리 사용량이 4GB 이상인 경우 경고
            if process_stats['memory_gb'] > 4.0:
                logger.warning("프로세스 메모리 사용량이 높습니다",
                               context={"memory_gb": f"{process_stats['memory_gb']:.2f}GB"})

            return process_stats

    @staticmethod
    def _collect_gpu_stats() -> List[Dict[str, Any]]:
        """GPU 통계 수집"""
        with TimingContext(gpu_logger.debug, "GPU 통계 수집"):
            gpu_stats = []
            for idx in range(torch.cuda.device_count()):
                gpu_info = {
                    "id": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "memory_allocated_gb": torch.cuda.memory_allocated(idx) / (1024 ** 3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3)
                }
                gpu_stats.append(gpu_info)

                gpu_logger.debug(f"GPU {idx} 통계",
                                 context={
                                     "name": gpu_info["name"],
                                     "memory_allocated_gb": f"{gpu_info['memory_allocated_gb']:.2f}GB",
                                     "memory_reserved_gb": f"{gpu_info['memory_reserved_gb']:.2f}GB"
                                 })

                # GPU 메모리 사용량이 90% 이상인 경우 경고
                if gpu_info['memory_allocated_gb'] / gpu_info['memory_reserved_gb'] > 0.9:
                    gpu_logger.warning(f"GPU {idx} 메모리 사용량이 높습니다",
                                       context={
                                           "memory_allocated_gb": f"{gpu_info['memory_allocated_gb']:.2f}GB",
                                           "memory_reserved_gb": f"{gpu_info['memory_reserved_gb']:.2f}GB"
                                       })

            return gpu_stats


# 활성 프로파일러 인스턴스 (싱글톤)
_active_profiler = None


def get_profiler() -> SystemProfiler:
    """
    시스템 프로파일러 인스턴스 가져오기 (싱글톤)

    Returns:
        SystemProfiler 인스턴스
    """
    global _active_profiler

    # 모니터링이 비활성화된 경우 더미 프로파일러 반환
    if not config.monitoring.enabled:
        logger.debug("모니터링이 비활성화되어 더미 프로파일러를 반환합니다")
        return _get_dummy_profiler()

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

        logger.info("시스템 프로파일러 초기화",
                    context={
                        "profile_interval": local_profiler_config.profile_interval,
                        "memory_profiling": local_profiler_config.memory_profiling,
                        "torch_profiling": local_profiler_config.torch_profiling
                    })

        _active_profiler = SystemProfiler(local_profiler_config)
        _active_profiler.start()

    return _active_profiler


# 더미 프로파일러 생성 함수 추가
def _get_dummy_profiler():
    """모니터링 비활성화 상태에서 사용할 더미 프로파일러 생성"""
    if not hasattr(get_profiler, '_dummy_profiler'):
        class DummyProfiler:
            def __init__(self):
                self.config = ProfilerConfig(enabled=False)
                logger.debug("더미 프로파일러가 생성되었습니다")

            def profile_function(self, func):
                return func

            async def get_memory_report(self):
                return {"status": "profiling_disabled"}

            async def get_function_stats(self, *args, **kwargs):
                return {"status": "profiling_disabled"}

            def start(self):
                logger.debug("더미 프로파일러 시작 요청 (무시됨)")
                pass

            def stop(self):
                logger.debug("더미 프로파일러 중지 요청 (무시됨)")
                pass

        get_profiler._dummy_profiler = DummyProfiler()

    return get_profiler._dummy_profiler


@with_logging_context(module="profiler.decorator")
def profile(func_):
    """
    함수 프로파일링 데코레이터

    Args:
        func_: 프로파일링할 함수

    Returns:
        프로파일링이 적용된 함수
    """
    # 모니터링이 비활성화된 경우 원본 함수 그대로 반환
    if not config.monitoring.enabled:
        return func_

    # 함수명 및 모듈 로깅
    function_logger.debug(f"함수 '{func_.__name__}' 프로파일링 데코레이터 적용됨",
                          context={"module": func_.__module__})

    return get_profiler().profile_function(func_)


# 컨텍스트 관리자를 통한 코드 블록 프로파일링
@contextlib.contextmanager
def profile_block(name: str):
    """
    코드 블록 프로파일링을 위한 컨텍스트 관리자

    Args:
        name: 프로파일링 블록 이름
    """
    # 모니터링이 비활성화된 경우 빈 컨텍스트 반환
    if not config.monitoring.enabled:
        yield
        return

    profiler_ = get_profiler()
    if not profiler_.config.enabled:
        yield
        return

    function_logger.debug(f"블록 '{name}' 프로파일링 시작")

    # TimingContext를 사용하면 코드가 더 깔끔해짐
    with TimingContext(None, f"Block {name}") as timing:
        try:
            yield
        finally:
            # 함수 프로파일로 기록
            key = f"block.{name}"
            if key not in profiler_.function_profiles:
                profiler_.function_profiles[key] = FunctionProfile(
                    function_name=name,
                    module_name="block"
                )

            profiler_.function_profiles[key].update(timing.duration)

            function_logger.debug(f"블록 '{name}' 프로파일링 완료",
                                  context={"duration": f"{timing.duration:.6f}s"})


# API 엔드포인트용 함수
async def get_profiler_memory_report() -> Dict[str, Any]:
    """
    API를 통한 메모리 보고서 가져오기

    Returns:
        현재 메모리 사용 보고서
    """
    memory_logger.info("API: 메모리 보고서 요청됨")
    profiler_ = get_profiler()
    report = await profiler_.get_memory_report()
    memory_logger.debug("API: 메모리 보고서 반환 완료")
    return report


async def get_profiler_function_stats() -> Dict[str, Any]:
    """
    API를 통한 함수 통계 가져오기

    Returns:
        함수 실행 시간 통계
    """
    function_logger.info("API: 함수 통계 요청됨")
    profiler_ = get_profiler()
    stats = await profiler_.get_function_stats()
    function_logger.debug("API: 함수 통계 반환 완료")
    return stats


async def get_profiler_system_stats() -> Dict[str, Any]:
    """
    API를 통한 시스템 상태 통계 가져오기

    Returns:
        시스템 상태 보고서
    """
    logger.info("API: 시스템 상태 통계 요청됨")
    stats = await SystemProfiler.get_system_stats()
    logger.debug("API: 시스템 상태 통계 반환 완료")
    return stats


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
            logger.info("프로파일링 시스템 초기화 중")
            profiler_ = get_profiler()
            logger.info("프로파일링 시스템이 초기화되었습니다",
                        context={
                            "profile_interval": profiler_.config.profile_interval,
                            "memory_profiling": profiler_.config.memory_profiling,
                            "torch_profiling": profiler_.config.torch_profiling
                        })
            return profiler_
        except Exception as err:
            logger.error(f"프로파일링 초기화 실패: {str(err)}",
                         context={"error_type": type(err).__name__},
                         exc_info=True)
            return None
    else:
        logger.info("프로파일링이 설정에서 비활성화되어 있습니다")
        return None


# 테스트 유틸리티 함수들
class ProfilerTestUtils:
    """프로파일러 테스트를 위한 유틸리티 클래스"""

    @staticmethod
    @with_logging_context(module="profiler.test")
    def run_cpu_test(iterations: int = 1000000) -> int:
        """CPU 부하 테스트 함수"""
        logger.info(f"CPU 부하 테스트 시작 (n={iterations})")

        with TimingContext(logger.debug, "CPU 부하 테스트") as timing:
            result = 0
            for idx in range(iterations):
                result += idx ** 2

            logger.info(f"CPU 부하 테스트 완료",
                        context={
                            "iterations": iterations,
                            "duration": f"{timing.duration:.3f}s"
                        })
            return result

    @staticmethod
    @with_logging_context(module="profiler.test")
    def run_memory_test(size_mb: int = 100) -> int:
        """메모리 사용량 테스트 함수"""
        memory_logger.info(f"메모리 테스트 시작 (size={size_mb}MB)")

        with TimingContext(memory_logger.debug, "메모리 할당 테스트") as timing:
            try:
                # size_mb 크기의 메모리 할당
                data = [0] * (size_mb * 131072)  # 약 1MB = 1024 * 1024 / 8 (bytes per int)
                result = sum(data[:100])  # 계산을 통해 최적화 방지

                memory_logger.info(f"메모리 테스트 완료",
                                   context={
                                       "size_mb": size_mb,
                                       "duration": f"{timing.duration:.3f}s"
                                   })
                return result
            except MemoryError as e:
                memory_logger.error(f"메모리 테스트 중 MemoryError 발생: {str(e)}",
                                    context={"requested_mb": size_mb},
                                    exc_info=True)
                return -1

    @staticmethod
    @with_logging_context(module="profiler.test")
    def run_gpu_test(tensor_size: int = 1000) -> float:
        """GPU 테스트 함수"""
        if not torch.cuda.is_available():
            gpu_logger.warning("CUDA를 사용할 수 없어 GPU 테스트를 건너뜁니다")
            return 0.0

        gpu_logger.info(f"GPU 테스트 시작 (크기: {tensor_size}x{tensor_size})")

        with TimingContext(gpu_logger.debug, "GPU 행렬 곱셈 테스트") as timing:
            try:
                x = torch.rand(tensor_size, tensor_size, device="cuda")
                y = torch.rand(tensor_size, tensor_size, device="cuda")

                # 행렬 곱셈 수행
                with profile_block("gpu_matrix_multiply"):
                    z = torch.matmul(x, y)
                result = z.sum().item()

                gpu_logger.info(f"GPU 테스트 완료",
                                context={
                                    "tensor_size": tensor_size,
                                    "duration": f"{timing.duration:.3f}s",
                                    "result": f"{result:.2f}"
                                })
                return result
            except Exception as e:
                gpu_logger.error(f"GPU 테스트 중 오류 발생: {str(e)}",
                                 context={"tensor_size": tensor_size},
                                 exc_info=True)
                return 0.0

    @staticmethod
    def visualize_results(t_profiler: SystemProfiler, t_output_dir: str) -> None:
        """프로파일링 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            logger.info("프로파일링 결과 시각화 시작")
            os.makedirs(t_output_dir, exist_ok=True)

            # 메모리 스냅샷 시각화
            with TimingContext(memory_logger.debug, "메모리 사용량 시각화"):
                ProfilerTestUtils._visualize_memory_usage(t_profiler, t_output_dir)

            # GPU 메모리 시각화
            with TimingContext(gpu_logger.debug, "GPU 메모리 시각화"):
                ProfilerTestUtils._visualize_gpu_memory(t_profiler, t_output_dir)

            # 함수 실행 시간 시각화
            with TimingContext(function_logger.debug, "함수 실행 시간 시각화"):
                ProfilerTestUtils._visualize_function_times(t_profiler, t_output_dir)

            logger.info(f"결과 시각화 완료: {t_output_dir}")

        except ImportError:
            logger.warning("matplotlib이 설치되지 않아 시각화를 건너뜁니다")
        except Exception as err:
            logger.error(f"시각화 중 오류 발생: {err}",
                         context={"error_type": type(err).__name__},
                         exc_info=True)

    @staticmethod
    def _visualize_memory_usage(t_profiler: SystemProfiler, t_output_dir: str) -> None:
        """메모리 사용량 시각화"""
        import matplotlib.pyplot as plt

        if len(t_profiler.memory_snapshots) <= 1:
            memory_logger.warning("메모리 스냅샷이 부족하여 시각화를 건너뜁니다")
            return

        # 시간에 따른 프로세스 메모리 사용량
        plt.figure(figsize=(10, 6))
        timestamps = list(range(len(t_profiler.memory_snapshots)))
        memory_values = [snap.process_ram_used_gb for snap in t_profiler.memory_snapshots]

        plt.plot(timestamps, memory_values, 'b-', marker='o')
        plt.title('프로세스 메모리 사용량 변화')
        plt.xlabel('스냅샷 순서')
        plt.ylabel('메모리 사용량 (GB)')
        plt.grid(True)

        output_file = f"{t_output_dir}/memory_usage.png"
        plt.savefig(output_file)
        memory_logger.info(f"메모리 사용량 그래프 저장됨: {output_file}")

    @staticmethod
    def _visualize_gpu_memory(t_profiler: SystemProfiler, t_output_dir: str) -> None:
        """GPU 메모리 사용량 시각화"""
        import matplotlib.pyplot as plt

        if not torch.cuda.is_available() or len(t_profiler.memory_snapshots) <= 1:
            gpu_logger.debug("GPU 또는 메모리 스냅샷이 부족하여 GPU 그래프를 생성하지 않습니다")
            return

        if not t_profiler.memory_snapshots[0].gpu_memory:
            gpu_logger.warning("GPU 메모리 정보가 없어 시각화를 건너뜁니다")
            return

        plt.figure(figsize=(10, 6))
        timestamps = list(range(len(t_profiler.memory_snapshots)))
        gpu_memory_values = []

        # 각 GPU별 메모리 사용량 추출
        for gpu_idx in range(torch.cuda.device_count()):
            values = []
            for snap in t_profiler.memory_snapshots:
                if gpu_idx < len(snap.gpu_memory):
                    values.append(snap.gpu_memory[gpu_idx].memory_allocated_gb)
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

        output_file = f"{t_output_dir}/gpu_memory_usage.png"
        plt.savefig(output_file)
        gpu_logger.info(f"GPU 메모리 그래프 저장됨: {output_file}")

    @staticmethod
    def _visualize_function_times(t_profiler: SystemProfiler, t_output_dir: str) -> None:
        """함수 실행 시간 시각화"""
        import matplotlib.pyplot as plt

        if not t_profiler.function_profiles:
            function_logger.warning("함수 프로파일이 없어 시각화를 건너뜁니다")
            return

        plt.figure(figsize=(12, 8))

        # 상위 10개 함수만 선택
        top_functions = sorted(
            t_profiler.function_profiles.values(),
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
        output_file = f"{t_output_dir}/function_times.png"
        plt.savefig(output_file)
        function_logger.info(f"함수 실행 시간 그래프 저장됨: {output_file}",
                             context={"function_count": len(function_names)})


# 모듈 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logger = get_logger("profiler-test")
    memory_logger = get_logger("profiler-test.memory")
    function_logger = get_logger("profiler-test.function")
    gpu_logger = get_logger("profiler-test.gpu")

    logger.info("프로파일러 테스트 시작")

    # 결과 저장 디렉토리 생성
    output_dir = f"./profile_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"결과 저장 디렉토리 생성됨: {output_dir}")

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
    logger.info("프로파일러가 시작되었습니다")

    try:
        # CPU 부하 테스트
        @profile
        def cpu_intensive_test(num):
            return ProfilerTestUtils.run_cpu_test(num)


        # 메모리 사용 테스트
        @profile
        def memory_test(size_mb):
            return ProfilerTestUtils.run_memory_test(size_mb)


        # 의도적인 메모리 누수 시뮬레이션
        memory_leak_storage = []


        @profile
        def simulate_memory_leak(size_mb, rounds=5):
            """의도적인 메모리 누수 시뮬레이션"""
            memory_logger.info(f"메모리 누수 시뮬레이션 시작",
                               context={"size_mb": size_mb, "rounds": rounds})

            for idx in range(rounds):
                # 전역 변수에 저장하여 참조가 유지되도록 함
                data = "X" * (size_mb * 1024 * 1024)  # size_mb MB
                memory_leak_storage.append(data)
                memory_logger.info(f"메모리 누수 라운드 완료",
                                   context={
                                       "round": f"{idx + 1}/{rounds}",
                                       "objects": len(memory_leak_storage)
                                   })
                time.sleep(1)  # 메모리 사용량 변화 관찰을 위한 지연


        # GPU 테스트
        @profile
        def gpu_test():
            """GPU 사용 테스트 (CUDA 가능한 경우)"""
            if not torch.cuda.is_available():
                gpu_logger.warning("CUDA를 사용할 수 없어 GPU 테스트를 건너뜁니다")
                return

            gpu_logger.info(f"GPU 테스트 시작 (장치: {torch.cuda.get_device_name(0)})")

            # 다양한 크기로 GPU 테스트 실행
            for size_t in [1000, 2000, 3000]:
                with TimingContext(gpu_logger.debug, f"GPU 테스트 (크기: {size_t})"):
                    ProfilerTestUtils.run_gpu_test(size_t)
                time.sleep(1)  # GPU 메모리 해제를 위한 지연


        # 동기 방식으로 시스템 상태를 가져오는 래퍼 함수
        def get_sync_system_stats():
            return asyncio.run(SystemProfiler.get_system_stats())


        def get_sync_memory_report():
            return asyncio.run(profiler.get_memory_report())


        def get_sync_function_stats():
            return asyncio.run(profiler.get_function_stats(top_n=5))


        # 테스트 실행
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

            logger.info("시스템 상태 모니터링",
                        context={
                            "iteration": i + 1,
                            "cpu_percent": f"{system_stats['cpu']['usage_percent']:.1f}%",
                            "ram_percent": f"{system_stats['memory']['percent']:.1f}%"
                        })
            time.sleep(2)

        # 결과 수집 및 보고서 생성
        logger.info("===== 결과 수집 중 =====")

        # 메모리 보고서
        memory_report = get_sync_memory_report()
        if "current" in memory_report:
            memory_logger.info("메모리 보고서 생성됨",
                               context={
                                   "process_ram_gb": f"{memory_report['current']['process_ram_used_gb']:.2f}GB"
                               })

        # 함수 통계
        function_stats = get_sync_function_stats()
        function_logger.info("함수 실행 시간 통계",
                             context={
                                 "count": len(function_stats.get('top_functions', [])),
                                 "top_functions": [
                                     {
                                         "name": f"{func['module_name']}.{func['function_name']}",
                                         "calls": func['call_count'],
                                         "avg_time": f"{func['avg_time']:.6f}s"
                                     } for func in function_stats.get('top_functions', [])[:3]  # 상위 3개만
                                 ]
                             })

        # 결과 JSON 저장
        result_data = {
            "memory_report": memory_report,
            "function_stats": function_stats,
            "system_stats": get_sync_system_stats()
        }

        result_file = f"{output_dir}/profile_results.json"
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"결과 JSON 저장됨: {result_file}")

        # 결과 시각화
        ProfilerTestUtils.visualize_results(profiler, output_dir)

        logger.info(f"===== 테스트 완료 =====")
        logger.info(f"모든 결과는 {output_dir} 디렉토리에 저장되었습니다")

    except KeyboardInterrupt:
        logger.info("사용자에 의한 테스트 중단")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}",
                     context={"error_type": type(e).__name__},
                     exc_info=True)
    finally:
        # 프로파일러 정리
        profiler.stop()
        logger.info("프로파일러가 종료되었습니다")

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
