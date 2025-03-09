"""
mai-vllm-serving 향상된 구조화 로깅 유틸리티
모든 모듈에서 일관된 JSON 구조화 로깅을 위한 기능 제공
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict, Any, Callable, List

# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config


# 로그 디렉토리 생성 함수
def ensure_log_directory(log_path: str) -> None:
    """
    로그 디렉토리가 존재하는지 확인하고, 없으면 생성

    Args:
        log_path: 로그 파일 경로
    """
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)


# 로그 회전 정책 클래스
class LogRotationPolicy:
    """로그 회전 및 보관 정책 관리 클래스"""

    def __init__(self,
                 base_log_dir: str = "./logs",
                 max_size_mb: int = 100,
                 backup_count: int = 30,
                 compression: bool = True,
                 retention_days: int = 90):
        """
        로그 회전 및 보관 정책 초기화

        Args:
            base_log_dir: 기본 로그 디렉토리
            max_size_mb: 로그 파일 최대 크기 (MB)
            backup_count: 유지할 백업 파일 수
            compression: 오래된 로그 파일 압축 여부
            retention_days: 로그 보관 기간 (일)
        """
        self.base_log_dir = base_log_dir
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
        self.compression = compression
        self.retention_days = retention_days

        # 로그 디렉토리 생성
        os.makedirs(self.base_log_dir, exist_ok=True)

        # 로거 가져오기
        self.logger = logging.getLogger(__name__)

    def apply_to_handler(self, handler: TimedRotatingFileHandler) -> None:
        """
        로그 핸들러에 회전 정책 적용

        Args:
            handler: 적용할 TimedRotatingFileHandler 인스턴스
        """
        handler.backupCount = self.backup_count
        # max_bytes는 TimedRotatingFileHandler에 없으므로 별도 처리 필요

    def cleanup_old_logs(self) -> Dict[str, Any]:
        """
        오래된 로그 파일 정리

        Returns:
            정리 결과 정보
        """
        if not os.path.exists(self.base_log_dir):
            return {"status": "skipped", "reason": "log_dir_not_exists"}

        result = {
            "deleted_count": 0,
            "compressed_count": 0,
            "total_size_saved_mb": 0,
            "retention_policy": f"{self.retention_days} days"
        }

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_timestamp = cutoff_date.timestamp()

        log_files = self._get_log_files()

        for log_file in log_files:
            full_path = os.path.join(self.base_log_dir, log_file)
            file_stat = os.stat(full_path)
            file_size_mb = file_stat.st_size / (1024 * 1024)

            # 만료된 로그 삭제
            if file_stat.st_mtime < cutoff_timestamp:
                try:
                    os.remove(full_path)
                    result["deleted_count"] += 1
                    result["total_size_saved_mb"] += file_size_mb
                    self.logger.debug(f"Deleted old log file: {log_file} ({file_size_mb:.2f} MB)")
                except Exception as e:
                    self.logger.warning(f"Failed to delete old log file {log_file}: {str(e)}")
                continue

            # 압축이 활성화된 경우 오래된 로그 압축
            if self.compression and not log_file.endswith('.gz') and '_' in log_file:
                # 현재 로그 파일은 압축하지 않음 (파일명에 날짜가 없는 경우)
                try:
                    compressed_path = full_path + '.gz'
                    with open(full_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # 원본 크기와 압축 후 크기 비교
                    compressed_size = os.stat(compressed_path).st_size / (1024 * 1024)
                    size_saved = file_size_mb - compressed_size

                    # 압축이 성공하면 원본 삭제
                    os.remove(full_path)
                    result["compressed_count"] += 1
                    result["total_size_saved_mb"] += size_saved
                    self.logger.debug(
                        f"Compressed log file: {log_file} "
                        f"(saved: {size_saved:.2f} MB, compression ratio: {compressed_size / file_size_mb:.2f})"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to compress log file {log_file}: {str(e)}")

        return result

    def enforce_size_limit(self) -> Dict[str, Any]:
        """
        로그 파일 크기 제한 적용
        현재 활성 로그 파일이 최대 크기를 초과하면 즉시 회전

        Returns:
            로그 회전 결과 정보
        """
        result = {
            "rotated_files": 0,
            "rotation_reason": "",
            "status": "no_action"
        }

        # 현재 활성 로그 파일들 확인
        active_logs = self._get_active_log_files()

        for log_file, full_path in active_logs:
            # 파일 크기 확인
            try:
                file_size_mb = os.path.getsize(full_path) / (1024 * 1024)

                # 최대 크기 초과 시 로그 회전
                if file_size_mb > self.max_size_mb:
                    self._rotate_log_file(log_file, full_path)
                    result["rotated_files"] += 1
                    result["status"] = "rotated"
                    result["rotation_reason"] = f"size_limit_exceeded ({file_size_mb:.2f}MB > {self.max_size_mb}MB)"
                    self.logger.info(
                        f"Rotated log file due to size limit: {log_file} ({file_size_mb:.2f}MB)"
                    )
            except Exception as e:
                self.logger.warning(f"Error checking/rotating log file {log_file}: {str(e)}")

        return result

    def _get_log_files(self) -> List[str]:
        """
        로그 디렉토리의 모든 로그 파일 목록 반환

        Returns:
            로그 파일명 리스트
        """
        if not os.path.exists(self.base_log_dir):
            return []

        return [f for f in os.listdir(self.base_log_dir)
                if os.path.isfile(os.path.join(self.base_log_dir, f)) and
                (f.endswith('.log') or f.endswith('.log.gz'))]

    def _get_active_log_files(self) -> List[tuple]:
        """
        현재 활성 상태인 로그 파일 목록 반환 (회전되지 않은 파일)

        Returns:
            (파일명, 전체 경로) 튜플 리스트
        """
        result = []

        for f in os.listdir(self.base_log_dir):
            full_path = os.path.join(self.base_log_dir, f)
            # 날짜 접미사가 없고 .gz로 끝나지 않는 .log 파일
            if (os.path.isfile(full_path) and
                    f.endswith('.log') and
                    not f.endswith('.gz') and
                    '_' not in f):  # 회전된 파일은 보통 파일명_날짜.log 형식
                result.append((f, full_path))

        return result

    def _rotate_log_file(self, log_file: str, full_path: str) -> bool:
        """
        로그 파일 수동 회전

        Args:
            log_file: 로그 파일명
            full_path: 로그 파일 전체 경로

        Returns:
            회전 성공 여부
        """
        try:
            # 현재 시간 기반 접미사 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 파일 이름과 확장자 분리
            name_parts = os.path.splitext(full_path)
            rotated_path = f"{name_parts[0]}_{timestamp}{name_parts[1]}"

            # 파일 회전
            shutil.copy2(full_path, rotated_path)

            # 원본 파일 비우기
            with open(full_path, 'w') as f:
                f.truncate(0)

            self.logger.info(f"Successfully rotated log file: {log_file} -> {os.path.basename(rotated_path)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to rotate log file {log_file}: {str(e)}")
            return False


# UTF-8 인코딩을 사용하는 고급 시간 기반 로그 로테이션 핸들러
class AdvancedRotatingFileHandler(TimedRotatingFileHandler):
    """
    고급 시간 기반 로그 회전 핸들러

    크기 제한 및 시간 기반 회전을 모두 지원하며 압축 기능 포함
    """

    def __init__(self,
                 filename: str,
                 when: str = 'midnight',
                 interval: int = 1,
                 backup_count: int = 30,
                 encoding: str = 'utf-8',
                 max_size_mb: int = 100,
                 compression: bool = True):
        """
        고급 로그 핸들러 초기화

        Args:
            filename: 로그 파일 경로
            when: 회전 시점 ('S', 'M', 'H', 'D', 'W0'-'W6', 'midnight')
            interval: 회전 간격
            backup_count: 유지할 백업 파일 수
            encoding: 파일 인코딩
            max_size_mb: 로그 파일 최대 크기 (MB)
            compression: 오래된 로그 파일 압축 여부
        """
        super().__init__(
            filename, when, interval, backup_count,
            encoding=encoding, delay=False, utc=False
        )
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression = compression
        self.logger = logging.getLogger(__name__)

    def emit(self, record: logging.LogRecord) -> None:
        """
        로그 레코드 출력 및 크기 기반 회전 처리

        Args:
            record: 로그 레코드
        """
        try:
            # 파일이 없는 경우 생성
            if self.stream is None:
                self.stream = self._open()

            # 현재 파일 크기 확인
            self.stream.seek(0, 2)  # 파일의 끝으로 이동
            current_size = self.stream.tell()

            # 크기 제한 초과 시 회전
            if current_size >= self.max_size_bytes:
                self.stream.close()
                self.stream = None
                self.doRollover()
                # 새 파일 열기
                if not self.stream:
                    self.stream = self._open()

            # 로그 레코드 처리
            super().emit(record)

        except Exception as e:
            self.handleError(record)
            self.logger.error(f"Error in AdvancedRotatingFileHandler.emit: {str(e)}")

    def doRollover(self) -> None:
        """로그 파일 회전 수행 및 압축 처리"""
        # 현재 시간 기반 접미사 생성 (YYYYMMDD_HHMMSS 형식)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일 이름과 확장자 분리
        base_filename, ext = os.path.splitext(self.baseFilename)
        new_filename = f"{base_filename}_{current_time}{ext}"

        # 이미 존재하는 경우 처리
        if os.path.exists(new_filename):
            i = 1
            while os.path.exists(f"{base_filename}_{current_time}_{i}{ext}"):
                i += 1
            new_filename = f"{base_filename}_{current_time}_{i}{ext}"

        # 파일 닫기
        if self.stream:
            self.stream.close()
            self.stream = None

        try:
            # 파일 이름 변경 대신 복사 후 원본 비우기
            shutil.copy2(self.baseFilename, new_filename)
            with open(self.baseFilename, 'w') as f:
                f.truncate(0)
        except Exception as e:
            self.logger.error(f"Error during log rotation: {str(e)}")

        # 새 파일 열기
        if not self.delay:
            self.stream = self._open()

        # 백업 파일 개수 유지
        if self.backupCount > 0:
            self._do_backup_count_maintenance()

        # 오래된 로그 파일 압축
        if self.compression:
            self._compress_old_logs()

    def _do_backup_count_maintenance(self) -> None:
        """백업 파일 개수 유지"""
        dir_name, base_name = os.path.split(self.baseFilename)
        base_base, base_ext = os.path.splitext(base_name)

        # 디렉토리 내 동일 기본 이름을 가진 백업 파일들 찾기
        backup_files = []
        for filename in os.listdir(dir_name):
            if filename.startswith(base_base) and filename != base_name:
                backup_files.append(os.path.join(dir_name, filename))

        # 오래된 파일 삭제
        if len(backup_files) > self.backupCount:
            backup_files.sort()  # 파일명으로 정렬 (날짜 기반 이름이므로 시간순)
            for old_file in backup_files[:-self.backupCount]:
                try:
                    os.remove(old_file)
                    self.logger.debug(f"Deleted old log file: {os.path.basename(old_file)}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete old log file {old_file}: {str(e)}")

    def _compress_old_logs(self) -> None:
        """오래된 로그 파일 압축"""
        log_dir = os.path.dirname(self.baseFilename)
        base_name = os.path.basename(self.baseFilename)

        for filename in os.listdir(log_dir):
            # 현재 활성 로그 파일은 건너뛰기
            if filename == base_name or filename.endswith('.gz'):
                continue

            # 현재 로그 파일의 회전된 버전인 경우에만 처리
            if filename.startswith(os.path.splitext(base_name)[0]) and filename.endswith('.log'):
                log_path = os.path.join(log_dir, filename)
                try:
                    # 압축된 파일 경로
                    gz_path = log_path + '.gz'

                    # 원본이 1일 이상 지난 경우에만 압축
                    file_mtime = os.path.getmtime(log_path)
                    if time.time() - file_mtime >= 86400:  # 24시간
                        with open(log_path, 'rb') as f_in:
                            with gzip.open(gz_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        # 압축 성공 시 원본 삭제
                        if os.path.exists(gz_path):
                            original_size = os.path.getsize(log_path)
                            compressed_size = os.path.getsize(gz_path)
                            compression_ratio = compressed_size / original_size if original_size > 0 else 0

                            os.remove(log_path)
                            self.logger.debug(
                                f"Compressed log file: {filename} "
                                f"(saved: {(original_size - compressed_size) / 1024:.1f} KB, "
                                f"ratio: {compression_ratio:.2f})"
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to compress log file {filename}: {str(e)}")


# JSON 로깅을 위한 사용자 정의 Formatter
class JsonFormatter(logging.Formatter):
    """향상된 JSON 형식의 로그 포맷터"""

    def __init__(self, service_name: str = "mai-vllm-serving", include_caller_info: bool = True):
        """
        JSON 로그 포맷터 초기화

        Args:
            service_name: 서비스 이름 (로그에 포함됨)
            include_caller_info: 호출자 정보(파일명, 라인번호) 포함 여부
        """
        super().__init__()
        self.service_name = service_name
        self.include_caller_info = include_caller_info

    def format(self, record: logging.LogRecord) -> str:
        """
        로그 레코드를 JSON 형식으로 포맷팅

        Args:
            record: 로그 레코드

        Returns:
            JSON 형식의 로그 문자열
        """
        # 기본 로그 정보
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
            "process": record.process
        }

        # 호출자 정보 추가 (선택적)
        if self.include_caller_info:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }

        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # 추가 속성이 있는 경우 추가
        if hasattr(record, "extra") and record.extra:
            # 충돌 방지를 위해 'extra' 키로 묶음
            log_data["extra"] = record.extra

        # 상세한 로깅 컨텍스트가 있는 경우
        if hasattr(record, "context") and record.context:
            log_data["context"] = record.context

        return json.dumps(log_data, ensure_ascii=False)


# 요청/응답 로깅을 위한 클래스
class RequestResponseLogger:
    """API 요청 및 응답 로깅을 위한 클래스"""

    def __init__(self, logger: logging.Logger):
        """
        요청/응답 로거 초기화

        Args:
            logger: 사용할 로거 인스턴스
        """
        self.logger = logger
        self.request_ids = {}  # 스레드 ID -> 요청 ID 매핑
        self.lock = threading.Lock()

    def log_request(self, request_data: Dict[str, Any],
                    request_id: Optional[str] = None,
                    client_ip: Optional[str] = None) -> str:
        """
        API 요청 로깅

        Args:
            request_data: 요청 데이터
            request_id: 요청 ID (없는 경우 생성)
            client_ip: 클라이언트 IP 주소

        Returns:
            요청 ID
        """
        # 요청 ID 생성 또는 사용
        if request_id is None:
            request_id = str(uuid.uuid4())

        # 스레드 ID와 요청 ID 매핑
        thread_id = threading.get_ident()
        with self.lock:
            self.request_ids[thread_id] = request_id

        # 로그 데이터 준비
        log_data = {
            "event": "request",
            "request_id": request_id,
            "client_ip": client_ip,
            "extra": {
                "request": self._sanitize_request(request_data)
            }
        }

        # 로깅
        self.logger.info(f"Request {request_id} received", extra=log_data)

        return request_id

    def log_response(self, response_data: Dict[str, Any],
                     status_code: int = 200,
                     processing_time: Optional[float] = None,
                     request_id: Optional[str] = None) -> None:
        """
        API 응답 로깅

        Args:
            response_data: 응답 데이터
            status_code: HTTP 상태 코드
            processing_time: 처리 시간 (초)
            request_id: 요청 ID (없는 경우 스레드에서 찾음)
        """
        # 스레드에서 요청 ID 찾기
        if request_id is None:
            thread_id = threading.get_ident()
            with self.lock:
                request_id = self.request_ids.get(thread_id, str(uuid.uuid4()))
                # 요청 ID 매핑 제거
                if thread_id in self.request_ids:
                    del self.request_ids[thread_id]

        # 로그 데이터 준비
        log_data = {
            "event": "response",
            "request_id": request_id,
            "status_code": status_code,
            "extra": {
                "response": self._sanitize_response(response_data),
                "processing_time": processing_time
            }
        }

        # 로깅
        self.logger.info(f"Response {request_id} sent with status {status_code}",
                         extra=log_data)

    def log_error(self, error: Exception,
                  request_data: Optional[Dict[str, Any]] = None,
                  status_code: int = 500,
                  request_id: Optional[str] = None) -> None:
        """
        API 오류 로깅

        Args:
            error: 발생한 예외
            request_data: 요청 데이터 (있는 경우)
            status_code: HTTP 상태 코드
            request_id: 요청 ID (없는 경우 스레드에서 찾음)
        """
        # 스레드에서 요청 ID 찾기
        if request_id is None:
            thread_id = threading.get_ident()
            with self.lock:
                request_id = self.request_ids.get(thread_id, str(uuid.uuid4()))
                # 요청 ID 매핑 제거
                if thread_id in self.request_ids:
                    del self.request_ids[thread_id]

        # 로그 데이터 준비
        log_data = {
            "event": "error",
            "request_id": request_id,
            "status_code": status_code,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "extra": {}
        }

        # 요청 데이터가 있는 경우 추가
        if request_data:
            log_data["extra"]["request"] = self._sanitize_request(request_data)

        # 로깅
        self.logger.error(f"Error processing request {request_id}: {str(error)}",
                          exc_info=True, extra=log_data)

    @classmethod
    def _sanitize_request(cls, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        요청 데이터 정제 (민감 정보 제거)

        Args:
            request_data: 요청 데이터

        Returns:
            정제된 요청 데이터
        """
        # 설정 객체에서 로깅 설정 가져오기
        config = get_config()

        # 민감 정보가 포함될 수 있는 필드 마스킹
        if not config.logging.log_requests:
            return {"masked": "Request logging disabled"}

        # 요청 데이터 복사
        sanitized = request_data.copy()

        # 토큰이나 인증 정보 마스킹
        if "token" in sanitized:
            sanitized["token"] = "****"
        if "api_key" in sanitized:
            sanitized["api_key"] = "****"
        if "authorization" in sanitized:
            sanitized["authorization"] = "****"

        return sanitized

    @classmethod
    def _sanitize_response(cls, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        응답 데이터 정제 (민감 정보 제거 및 대용량 응답 축약)

        Args:
            response_data: 응답 데이터

        Returns:
            정제된 응답 데이터
        """
        # 설정 객체에서 로깅 설정 가져오기
        config = get_config()

        # 응답 로깅이 비활성화된 경우
        if not config.logging.log_responses:
            return {"masked": "Response logging disabled"}

        # 응답 데이터 복사
        sanitized = response_data.copy()

        # 생성된 텍스트가 있는 경우 축약
        if "generated_text" in sanitized and isinstance(sanitized["generated_text"], str):
            if len(sanitized["generated_text"]) > 100:
                sanitized["generated_text"] = (
                        sanitized["generated_text"][:100] + "... (truncated)"
                )

        return sanitized


# 컨텍스트 관리자를 통한 요청 처리 시간 측정
class TimingContext:
    """요청 처리 시간 측정을 위한 컨텍스트 관리자"""

    def __init__(self, logger: Optional[logging.Logger] = None, label: str = "Operation"):
        """
        타이밍 컨텍스트 초기화

        Args:
            logger: 로거 인스턴스 (None인 경우 로깅하지 않음)
            label: 작업 레이블
        """
        self.logger = logger
        self.label = label
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """컨텍스트 시작 시 시간 기록"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 시간 기록 및 로깅"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if self.logger:
            self.logger.info(f"{self.label} completed in {duration:.3f} seconds")

        return False  # 예외를 전파함

    @property
    def duration(self):
        """작업 처리 시간 (초)"""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time


# 구조화된 로깅을 위한 래퍼 클래스
class StructuredLogger:
    """구조화된 로깅을 위한 래퍼 클래스"""

    def __init__(self, logger: logging.Logger):
        """
        구조화된 로거 초기화

        Args:
            logger: 기본 로거 인스턴스
        """
        self.logger = logger
        self.thread_local = threading.local()

        # 기본 컨텍스트 정보
        self.thread_local.context = {}

        # 요청 ID 추적 (각 스레드마다 별도 저장)
        self.thread_local.request_id = None

    def with_context(self, **kwargs) -> 'StructuredLogger':
        """
        로깅 컨텍스트 설정

        Args:
            **kwargs: 컨텍스트에 추가할 키-값 쌍

        Returns:
            현재 구조화된 로거 인스턴스
        """
        if not hasattr(self.thread_local, 'context'):
            self.thread_local.context = {}

        self.thread_local.context.update(kwargs)
        return self

    def clear_context(self) -> None:
        """로깅 컨텍스트 초기화"""
        self.thread_local.context = {}

    def set_request_id(self, request_id: str) -> None:
        """
        현재 스레드의 요청 ID 설정

        Args:
            request_id: 요청 ID
        """
        self.thread_local.request_id = request_id

    def get_request_id(self) -> Optional[str]:
        """
        현재 스레드의 요청 ID 반환

        Returns:
            요청 ID 또는 None
        """
        return getattr(self.thread_local, 'request_id', None)

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """
        내부 로깅 메서드

        Args:
            level: 로그 레벨
            msg: 로그 메시지
            *args: 포맷 인자
            **kwargs: 추가 인자
        """
        # extra 인자 처리
        extra = kwargs.pop('extra', {})

        # 컨텍스트 정보와 요청 ID 추가
        context = getattr(self.thread_local, 'context', {}).copy()

        request_id = getattr(self.thread_local, 'request_id', None)
        if request_id:
            context['request_id'] = request_id

        # kwargs에서 context 키워드가 있으면 기존 컨텍스트와 병합
        if 'context' in kwargs:
            context.update(kwargs.pop('context'))

        # extra에 컨텍스트 정보 추가
        extra['context'] = context

        # 원래 로거에 전달
        self.logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """DEBUG 레벨 로깅"""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """INFO 레벨 로깅"""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """WARNING 레벨 로깅"""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """ERROR 레벨 로깅"""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """CRITICAL 레벨 로깅"""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """예외 로깅 (ERROR 레벨, 스택 트레이스 포함)"""
        kwargs['exc_info'] = kwargs.get('exc_info', True)
        self._log(logging.ERROR, msg, *args, **kwargs)

    def log_api_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> None:
        """
        API 요청 로깅

        Args:
            method: HTTP 메서드
            endpoint: API 엔드포인트
            params: 요청 파라미터 (선택적)
        """
        self.info(f"API Request: {method} {endpoint}",
                  context={
                      'api': {
                          'method': method,
                          'endpoint': endpoint,
                          'params': params or {}
                      }
                  })

    def log_api_response(self, status_code: int, response_time: float, data: Optional[Any] = None) -> None:
        """
        API 응답 로깅

        Args:
            status_code: HTTP 상태 코드
            response_time: 응답 시간 (초)
            data: 응답 데이터 (선택적)
        """
        self.info(f"API Response: {status_code} in {response_time:.3f}s",
                  context={
                      'api': {
                          'status_code': status_code,
                          'response_time': response_time,
                          'data': data or {}
                      }
                  })

    def log_model_inference(self, model_name: str, tokens_in: int, tokens_out: int,
                            inference_time: float, tokens_per_second: float) -> None:
        """
        모델 추론 로깅

        Args:
            model_name: 모델 이름
            tokens_in: 입력 토큰 수
            tokens_out: 출력 토큰 수
            inference_time: 추론 시간 (초)
            tokens_per_second: 초당 생성 토큰 수
        """
        self.info(f"Model inference: {tokens_out} tokens in {inference_time:.3f}s ({tokens_per_second:.2f} tokens/sec)",
                  context={
                      'model': {
                          'name': model_name,
                          'tokens_in': tokens_in,
                          'tokens_out': tokens_out,
                          'inference_time': inference_time,
                          'tokens_per_second': tokens_per_second
                      }
                  })


def with_logging_context(func: Optional[Callable] = None, **context_kwargs):
    """
    로깅 컨텍스트를 설정하는 데코레이터

    Args:
        func: 데코레이트할 함수
        **context_kwargs: 로깅 컨텍스트에 설정할 키-값 쌍

    Returns:
        데코레이트된 함수
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 로거 이름은 함수가 속한 모듈에서 가져옴
            logger_name = f.__module__
            logger = get_logger(logger_name)

            # 함수 시작 시 컨텍스트 설정
            logger.with_context(function=f.__name__, **context_kwargs)

            # 함수 실행 시간 측정 시작
            start_time = time.time()

            try:
                # 실제 함수 실행
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                # 예외 발생 시 로깅
                logger.exception(f"Exception in {f.__name__}: {str(e)}")
                raise
            finally:
                # 실행 시간 로깅
                execution_time = time.time() - start_time
                logger.debug(f"Function {f.__name__} executed in {execution_time:.3f}s")

                # 컨텍스트 정리
                logger.clear_context()

        return wrapper

    # 데코레이터가 인자 없이 직접 사용된 경우
    if func is not None:
        return decorator(func)

    # 데코레이터가 인자와 함께 사용된 경우
    return decorator


def with_request_context(func: Optional[Callable] = None, request_id_arg: str = 'request_id'):
    """
    요청 ID를 로깅 컨텍스트에 설정하는 데코레이터

    Args:
        func: 데코레이트할 함수
        request_id_arg: 요청 ID를 포함하는 인자 이름

    Returns:
        데코레이트된 함수
    """

    def decorator(f):
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            # 로거 이름은 함수가 속한 모듈에서 가져옴
            logger_name = f.__module__
            logger = get_logger(logger_name)

            # 요청 ID 설정
            request_id = kwargs.get(request_id_arg)
            if request_id is None and len(args) > 0:
                # 첫 번째 인자가 RequestConfig인 경우
                first_arg = args[0]
                if hasattr(first_arg, 'request_id'):
                    request_id = first_arg.request_id

            # 요청 ID가 없으면 생성
            if request_id is None:
                request_id = str(uuid.uuid4())

            logger.set_request_id(request_id)
            logger.info(f"Request {request_id} started")

            # 시작 시간 기록
            start_time = time.time()

            try:
                # 비동기 함수 실행
                result = await f(*args, **kwargs)
                return result
            except Exception as e:
                # 예외 발생 시 로깅
                logger.exception(f"Request {request_id} failed: {str(e)}")
                raise
            finally:
                # 실행 시간 로깅
                execution_time = time.time() - start_time
                logger.info(f"Request {request_id} completed in {execution_time:.3f}s")

                # 요청 ID 정리
                logger.set_request_id(None)

        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            # 로거 이름은 함수가 속한 모듈에서 가져옴
            logger_name = f.__module__
            logger = get_logger(logger_name)

            # 요청 ID 설정
            request_id = kwargs.get(request_id_arg)
            if request_id is None and len(args) > 0:
                # 첫 번째 인자가 RequestConfig인 경우
                first_arg = args[0]
                if hasattr(first_arg, 'request_id'):
                    request_id = first_arg.request_id

            # 요청 ID가 없으면 생성
            if request_id is None:
                request_id = str(uuid.uuid4())

            logger.set_request_id(request_id)
            logger.info(f"Request {request_id} started")

            # 시작 시간 기록
            start_time = time.time()

            try:
                # 함수 실행
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                # 예외 발생 시 로깅
                logger.exception(f"Request {request_id} failed: {str(e)}")
                raise
            finally:
                # 실행 시간 로깅
                execution_time = time.time() - start_time
                logger.info(f"Request {request_id} completed in {execution_time:.3f}s")

                # 요청 ID 정리
                logger.set_request_id(None)

        # 비동기 함수인 경우 async_wrapper 반환, 그렇지 않으면 sync_wrapper 반환
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper

    # 데코레이터가 인자 없이 직접 사용된 경우
    if func is not None:
        return decorator(func)

    # 데코레이터가 인자와 함께 사용된 경우
    return decorator


# 로그 정리 스케줄러 함수
def schedule_log_cleanup(interval_hours: int = 24):
    """
    로그 정리 작업 스케줄링

    Args:
        interval_hours: 정리 작업 실행 간격 (시간)
    """
    import threading

    logger = logging.getLogger(__name__)
    logger.info(f"Starting log cleanup scheduler (interval: {interval_hours} hours)")

    def cleanup_task():
        while True:
            try:
                # 설정에서 로그 경로 가져오기
                config = get_config()
                log_file = config.logging.file or "./logs/mai_vllm_serving.log"
                log_dir = os.path.dirname(log_file)

                # 로그 정책 가져오기
                max_size_mb = getattr(config.logging, 'max_log_size_mb', 100)
                backup_count = getattr(config.logging, 'log_backup_count', 30)
                compression = getattr(config.logging, 'log_compression', True)
                retention_days = getattr(config.logging, 'log_retention_days', 90)

                # 로그 정책 객체 생성
                policy = LogRotationPolicy(
                    base_log_dir=log_dir,
                    max_size_mb=max_size_mb,
                    backup_count=backup_count,
                    compression=compression,
                    retention_days=retention_days
                )

                # 로그 정리 실행
                cleanup_result = policy.cleanup_old_logs()
                size_result = policy.enforce_size_limit()

                logger.info(
                    f"Scheduled log cleanup completed: "
                    f"deleted {cleanup_result['deleted_count']} files, "
                    f"compressed {cleanup_result['compressed_count']} files, "
                    f"saved {cleanup_result['total_size_saved_mb']:.2f} MB; "
                    f"Size check: {size_result['status']}"
                )

                # 다음 실행까지 대기
                time.sleep(interval_hours * 3600)
            except Exception as e:
                logger.error(f"Error in log cleanup task: {str(e)}")
                # 오류 발생 시 30분 후 재시도
                time.sleep(1800)

    # 백그라운드 스레드로 실행
    cleanup_thread = threading.Thread(
        target=cleanup_task,
        daemon=True,
        name="LogCleanupThread"
    )
    cleanup_thread.start()


# 로깅 초기화 함수
def setup_logging(service_name: str = "mai-vllm-serving",
                  log_level: Optional[str] = None,
                  log_format: Optional[str] = None,
                  log_file: Optional[str] = None,
                  use_json: Optional[bool] = None,
                  include_caller_info: bool = True,
                  max_size_mb: int = 100,
                  backup_count: int = 30,
                  compression: bool = True) -> logging.Logger:
    """
    향상된 로깅 시스템 초기화

    Args:
        service_name: 서비스 이름
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 로그 포맷 문자열
        log_file: 로그 파일 경로
        use_json: JSON 형식 로깅 사용 여부
        include_caller_info: 호출자 정보(파일명, 라인번호) 포함 여부
        max_size_mb: 로그 파일 최대 크기 (MB)
        backup_count: 유지할 백업 파일 수
        compression: 압축 활성화 여부

    Returns:
        설정된 로거 인스턴스
    """
    # 설정 객체에서 로깅 설정 가져오기
    config = get_config()

    # 인자가 None인 경우 설정에서 가져오기
    if log_level is None:
        log_level = config.logging.level
    if log_format is None:
        log_format = config.logging.format
    if log_file is None:
        log_file = config.logging.file
        # 기본 로그 파일 경로 설정
        if log_file is None:
            log_file = "./logs/mai_vllm_serving.log"
    if use_json is None:
        use_json = config.logging.json

    # 추가 로그 설정 가져오기
    if hasattr(config.logging, 'max_log_size_mb'):
        max_size_mb = config.logging.max_log_size_mb
    if hasattr(config.logging, 'log_backup_count'):
        backup_count = config.logging.log_backup_count
    if hasattr(config.logging, 'log_compression'):
        compression = config.logging.log_compression

    # 루트 로거 가져오기
    logger = logging.getLogger()

    # 로깅 레벨 설정
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 로그 포맷터 생성
    if use_json:
        formatter = JsonFormatter(service_name, include_caller_info)
    else:
        formatter = logging.Formatter(log_format, datefmt=config.logging.datefmt)

    # 로그 파일이 지정된 경우 고급 파일 핸들러 추가
    if log_file:
        # 로그 디렉토리 생성
        ensure_log_directory(log_file)

        # 고급 일별 로테이션 파일 핸들러 생성
        file_handler = AdvancedRotatingFileHandler(
            filename=log_file,
            when='midnight',  # 매일 자정에 로테이션
            interval=1,  # 1일마다
            backup_count=backup_count,  # 백업 파일 수
            encoding='utf-8',  # UTF-8 인코딩 사용
            max_size_mb=max_size_mb,  # 최대 크기
            compression=compression  # 압축 여부
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 로깅 설정 정보 출력
    logger.info(f"Logging initialized for {service_name}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log format: {'JSON' if use_json else 'Text'}")
    if log_file:
        logger.info(f"Log file: {log_file} (rotation: daily, size limit: {max_size_mb}MB, backups: {backup_count})")

    # 로그 정리 스케줄러 시작
    schedule_log_cleanup()

    return logger


# 요청/응답 로거 인스턴스 생성
def get_request_logger() -> RequestResponseLogger:
    """
    요청/응답 로거 인스턴스 가져오기

    Returns:
        RequestResponseLogger 인스턴스
    """
    logger = logging.getLogger("mai-vllm-serving.api")
    return RequestResponseLogger(logger)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    구조화된 로거 인스턴스 가져오기

    Args:
        name: 로거 이름

    Returns:
        StructuredLogger 인스턴스
    """
    logger = logging.getLogger(name)
    return StructuredLogger(logger)


# 모듈 레벨 싱글톤 구조화 로거 인스턴스
_structured_loggers = {}


def get_logger(name: str) -> StructuredLogger:
    """
    구조화된 로거 인스턴스 가져오기 (싱글톤)

    Args:
        name: 로거 이름

    Returns:
        StructuredLogger 인스턴스
    """
    global _structured_loggers

    if name not in _structured_loggers:
        _structured_loggers[name] = get_structured_logger(name)

    return _structured_loggers[name]


# 로그 정리 기능
def cleanup_logs(log_dir: Optional[str] = None,
                 retention_days: int = 90,
                 dry_run: bool = False) -> Dict[str, Any]:
    """
    로그 파일 정리 유틸리티 함수

    Args:
        log_dir: 로그 디렉토리 (None인 경우 설정에서 가져옴)
        retention_days: 보관 기간 (일)
        dry_run: 테스트 모드 (실제 삭제하지 않음)

    Returns:
        정리 결과 정보
    """
    # 로그 디렉토리 결정
    if log_dir is None:
        config = get_config()
        log_file = config.logging.file or "./logs/mai_vllm_serving.log"
        log_dir = os.path.dirname(log_file)

    if not os.path.exists(log_dir):
        return {
            "status": "skipped",
            "reason": "log_dir_not_exists",
            "log_dir": log_dir
        }

    logger = logging.getLogger(__name__)
    result = {
        "status": "completed" if not dry_run else "dry_run",
        "log_dir": log_dir,
        "retention_days": retention_days,
        "deleted_files": [],
        "compressed_files": [],
        "stats": {
            "total_files_checked": 0,
            "deleted_count": 0,
            "compressed_count": 0,
            "space_saved_bytes": 0
        }
    }

    # 기준일 계산
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_timestamp = cutoff_date.timestamp()

    # 로그 디렉토리 내 모든 파일 검사
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)

        # 디렉토리는 무시
        if not os.path.isfile(file_path):
            continue

        # 로그 파일만 처리 (.log 또는 .log.gz)
        if not (filename.endswith('.log') or filename.endswith('.log.gz')):
            continue

        result["stats"]["total_files_checked"] += 1

        try:
            file_stat = os.stat(file_path)
            file_mtime = file_stat.st_mtime
            file_size = file_stat.st_size

            # 보관 기간 초과 파일 삭제
            if file_mtime < cutoff_timestamp:
                if not dry_run:
                    os.remove(file_path)
                result["deleted_files"].append({
                    "name": filename,
                    "size_bytes": file_size,
                    "modified": datetime.fromtimestamp(file_mtime).isoformat()
                })
                result["stats"]["deleted_count"] += 1
                result["stats"]["space_saved_bytes"] += file_size
                logger.info(
                    f"Deleted old log file: {filename} (age: {(datetime.now().timestamp() - file_mtime) / 86400:.1f} days)")

            # 압축되지 않은 오래된 로그 파일 압축
            elif (not filename.endswith('.gz') and
                  '_' in filename and  # 회전된 파일 표시
                  (datetime.now().timestamp() - file_mtime) >= 86400):  # 1일 이상 경과

                if not dry_run:
                    compressed_path = file_path + '.gz'
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # 압축 성공하면 원본 삭제
                    if os.path.exists(compressed_path):
                        compressed_size = os.path.getsize(compressed_path)
                        space_saved = file_size - compressed_size

                        # 원본 삭제
                        os.remove(file_path)

                        # 통계 업데이트
                        result["stats"]["space_saved_bytes"] += space_saved
                        result["stats"]["compressed_count"] += 1
                        result["compressed_files"].append({
                            "name": filename,
                            "original_size_bytes": file_size,
                            "compressed_size_bytes": compressed_size,
                            "compression_ratio": compressed_size / file_size
                        })
                        logger.info(
                            f"Compressed log file: {filename} -> {filename}.gz "
                            f"(saved: {space_saved / 1024:.1f} KB, ratio: {compressed_size / file_size:.2f})"
                        )
                else:
                    # 드라이 런 모드에서는 추정치 계산
                    estimated_compressed_size = int(file_size * 0.3)  # 대략 70% 압축률 가정
                    estimated_space_saved = file_size - estimated_compressed_size

                    result["stats"]["space_saved_bytes"] += estimated_space_saved
                    result["stats"]["compressed_count"] += 1
                    result["compressed_files"].append({
                        "name": filename,
                        "original_size_bytes": file_size,
                        "estimated_compressed_size_bytes": estimated_compressed_size,
                        "estimated_compression_ratio": 0.3
                    })

        except Exception as e:
            logger.warning(f"Error processing log file {filename}: {str(e)}")

    # 바이트를 KB/MB로 변환
    result["stats"]["space_saved_kb"] = result["stats"]["space_saved_bytes"] / 1024
    result["stats"]["space_saved_mb"] = result["stats"]["space_saved_bytes"] / (1024 * 1024)

    return result


# 설정 정보를 로그 회전 정책에 반영하는 함수
def get_log_rotation_policy_from_config() -> LogRotationPolicy:
    """
    설정에서 로그 회전 정책 생성

    Returns:
        로그 회전 정책 객체
    """
    # 설정에서 로그 디렉토리 가져오기
    config = get_config()
    log_file = config.logging.file or "./logs/mai_vllm_serving.log"
    log_dir = os.path.dirname(log_file)

    # 기본값 설정
    max_size_mb = getattr(config.logging, 'max_log_size_mb', 100)
    backup_count = getattr(config.logging, 'log_backup_count', 30)
    compression = getattr(config.logging, 'log_compression', True)
    retention_days = getattr(config.logging, 'log_retention_days', 90)

    return LogRotationPolicy(
        base_log_dir=log_dir,
        max_size_mb=max_size_mb,
        backup_count=backup_count,
        compression=compression,
        retention_days=retention_days
    )


# 로그 관련 CLI 유틸리티 함수
def log_cleanup_cli(args=None):
    """
    로그 정리를 위한 명령행 유틸리티

    Args:
        args: 명령행 인자
    """
    import argparse

    parser = argparse.ArgumentParser(description="mai-vllm-serving 로그 정리 유틸리티")
    parser.add_argument("--log-dir", help="로그 디렉토리 경로 (지정하지 않으면 설정에서 가져옴)")
    parser.add_argument("--retention-days", type=int, default=90, help="로그 보관 기간 (일)")
    parser.add_argument("--dry-run", action="store_true", help="테스트 모드 (실제로 파일을 삭제하지 않음)")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")

    args = parser.parse_args(args)

    # 로깅 설정
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger("log_cleanup")
    logger.info(f"로그 정리 시작 (보관 기간: {args.retention_days}일, 드라이런: {args.dry_run})")

    # 로그 정리 실행
    result = cleanup_logs(
        log_dir=args.log_dir,
        retention_days=args.retention_days,
        dry_run=args.dry_run
    )

    # 결과 출력
    if result["status"] == "skipped":
        logger.warning(f"로그 디렉토리가 존재하지 않아 정리를 건너뜁니다: {result['log_dir']}")
    else:
        stats = result["stats"]
        logger.info(
            f"로그 정리 완료: "
            f"검사한 파일 {stats['total_files_checked']}개, "
            f"삭제한 파일 {stats['deleted_count']}개, "
            f"압축한 파일 {stats['compressed_count']}개, "
            f"절약한 용량 {stats['space_saved_mb']:.2f}MB"
        )

        if args.verbose:
            # 삭제된 파일 목록
            if result["deleted_files"]:
                logger.debug("삭제된 파일:")
                for file_info in result["deleted_files"]:
                    logger.debug(f"  - {file_info['name']} ({file_info['size_bytes'] / 1024:.1f}KB)")

            # 압축된 파일 목록
            if result["compressed_files"]:
                logger.debug("압축된 파일:")
                for file_info in result["compressed_files"]:
                    if "compressed_size_bytes" in file_info:
                        logger.debug(
                            f"  - {file_info['name']} "
                            f"({file_info['original_size_bytes'] / 1024:.1f}KB → "
                            f"{file_info['compressed_size_bytes'] / 1024:.1f}KB, "
                            f"비율: {file_info['compression_ratio']:.2f})"
                        )
                    else:
                        logger.debug(
                            f"  - {file_info['name']} "
                            f"({file_info['original_size_bytes'] / 1024:.1f}KB → "
                            f"추정 {file_info['estimated_compressed_size_bytes'] / 1024:.1f}KB, "
                            f"추정 비율: {file_info['estimated_compression_ratio']:.2f})"
                        )

    return 0


# 로깅 설정 문자열 변환 함수
def parse_log_level(level_str: str) -> int:
    """
    로그 레벨 문자열을 로깅 모듈 상수로 변환

    Args:
        level_str: 로그 레벨 문자열 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        로깅 모듈 레벨 상수
    """
    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level_str = level_str.lower()
    if level_str in level_map:
        return level_map[level_str]

    # 기본값은 INFO
    return logging.INFO


# 설정 파일에 로그 회전 설정 추가 함수
def update_config_with_log_rotation_settings(config_file: str = None) -> bool:
    """
    설정 파일에 로그 회전 관련 설정 추가

    Args:
        config_file: 설정 파일 경로 (None인 경우 기본 설정 파일 사용)

    Returns:
        성공 여부
    """
    import yaml
    from mai_vllm_serving.utils.config import DEFAULT_CONFIG_PATH

    # 설정 파일 경로 결정
    if config_file is None:
        config_file = DEFAULT_CONFIG_PATH

    try:
        # 설정 파일 읽기
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 로깅 섹션이 없으면 추가
        if 'logging' not in config_data:
            config_data['logging'] = {}

        # 로그 회전 설정 추가
        logging_config = config_data['logging']
        was_updated = False

        if 'max_log_size_mb' not in logging_config:
            logging_config['max_log_size_mb'] = 100
            was_updated = True

        if 'log_backup_count' not in logging_config:
            logging_config['log_backup_count'] = 30
            was_updated = True

        if 'log_compression' not in logging_config:
            logging_config['log_compression'] = True
            was_updated = True

        if 'log_retention_days' not in logging_config:
            logging_config['log_retention_days'] = 90
            was_updated = True

        # 변경된 경우에만 저장
        if was_updated:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            logger = logging.getLogger(__name__)
            logger.info(f"설정 파일 {config_file}에 로그 회전 설정이 추가되었습니다")
            return True

        return False

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"설정 파일 {config_file} 업데이트 중 오류 발생: {str(e)}")
        return False


# 명령행에서 직접 실행된 경우
if __name__ == "__main__":
    import sys

    # 간단한 명령행 인자 처리
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        # 로그 정리 명령 실행
        sys.exit(log_cleanup_cli(sys.argv[2:]))

    elif len(sys.argv) > 1 and sys.argv[1] == "update-config":
        # 설정 파일 업데이트 명령 실행
        config_file = sys.argv[2] if len(sys.argv) > 2 else None
        success = update_config_with_log_rotation_settings(config_file)
        sys.exit(0 if success else 1)

    else:
        # 도움말 출력
        print("사용법:")
        print("  python -m mai_vllm_serving.utils.logging_utils cleanup [options]")
        print("    로그 파일 정리 실행")
        print("    옵션:")
        print("      --log-dir PATH       로그 디렉토리 지정")
        print("      --retention-days N   로그 보관 기간 지정 (일)")
        print("      --dry-run            테스트 모드 (실제 삭제하지 않음)")
        print("      --verbose            상세 출력")
        print()
        print("  python -m mai_vllm_serving.utils.logging_utils update-config [CONFIG_FILE]")
        print("    설정 파일에 로그 회전 설정 추가")
        sys.exit(1)
