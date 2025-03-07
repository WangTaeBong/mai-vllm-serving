#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mai-vllm-serving 로깅 유틸리티
로깅 설정 및 관리를 위한 유틸리티 함수
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict, Any

# 설정 모듈 임포트
from mai_vllm_serving.utils.config import get_config


# locale 설정 제거하고 UTF-8 인코딩 사용
# locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8') 라인 제거


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


# UTF-8 인코딩을 사용하는 TimedRotatingFileHandler
class UTF8TimedRotatingFileHandler(TimedRotatingFileHandler):
    """UTF-8 인코딩을 사용하는 시간 기반 로그 로테이션 핸들러"""

    def __init__(self, filename, when='midnight', interval=1, backup_count=0,
                 encoding='utf-8', delay=False, utc=False, at_time=None):
        """
        UTF-8 인코딩을 강제하는 TimedRotatingFileHandler 초기화
        """
        super().__init__(
            filename, when, interval, backup_count,
            encoding=encoding, delay=delay, utc=utc, atTime=at_time
        )

    def doRollover(self):
        """
        로그 로테이션 수행 시 파일명에 날짜 추가
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # 현재 시간 기반 접미사 생성 (YYYYMMDD 형식)
        current_time = datetime.now().strftime("%Y%m%d")

        # 파일 이름과 확장자 분리
        if self.backupCount > 0:
            base_filename, ext = os.path.splitext(self.baseFilename)
            new_filename = f"{base_filename}_{current_time}{ext}"

            # 이미 존재하는 경우 처리
            if os.path.exists(new_filename):
                i = 1
                while os.path.exists(f"{base_filename}_{current_time}_{i}{ext}"):
                    i += 1
                new_filename = f"{base_filename}_{current_time}_{i}{ext}"

            try:
                # 파일 이름 변경
                os.rename(self.baseFilename, new_filename)
            except OSError:
                pass

        # 새 파일 열기
        if not self.delay:
            self.stream = self._open()


# JSON 로깅을 위한 사용자 정의 Formatter
class JsonFormatter(logging.Formatter):
    """JSON 형식의 로그 포맷터"""

    def __init__(self, service_name: str = "mai-vllm-serving"):
        """
        JSON 로그 포맷터 초기화

        Args:
            service_name: 서비스 이름 (로그에 포함됨)
        """
        super().__init__()
        self.service_name = service_name

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

        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # 추가 속성이 있는 경우 추가
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)

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


# 로깅 초기화 함수
def setup_logging(service_name: str = "mai-vllm-serving",
                  log_level: Optional[str] = None,
                  log_format: Optional[str] = None,
                  log_file: Optional[str] = None,
                  use_json: Optional[bool] = None) -> logging.Logger:
    """
    로깅 시스템 초기화

    Args:
        service_name: 서비스 이름
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 로그 포맷 문자열
        log_file: 로그 파일 경로
        use_json: JSON 형식 로깅 사용 여부

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
            log_file = "./logs/mai_serving.log"
    if use_json is None:
        use_json = config.logging.json

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
        formatter = JsonFormatter(service_name)
    else:
        formatter = logging.Formatter(log_format)

    # 로그 파일이 지정된 경우 파일 핸들러 추가
    if log_file:
        # 로그 디렉토리 생성
        ensure_log_directory(log_file)

        # 일별 로테이션 파일 핸들러 생성
        file_handler = UTF8TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',  # 매일 자정에 로테이션
            interval=1,  # 1일마다
            backup_count=30,  # 최대 30일치 보관
            encoding='utf-8'  # UTF-8 인코딩 사용
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
        logger.info(f"Log file: {log_file} (daily rotation enabled)")

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
