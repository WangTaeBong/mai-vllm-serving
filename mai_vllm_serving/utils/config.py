#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mai-vllm-serving 설정 관리 모듈
YAML 설정 파일을 로드하고 관리하는 기능 제공
"""

import argparse
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import yaml

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 설정 파일 경로
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "default_config.yaml"
)


@dataclass
class ServerConfig:
    """서버 설정 클래스"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    request_timeout: int = 300
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_batch_size: int = 32


@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    name: str = "meta-llama/Llama-2-7b-chat-hf"
    revision: str = "main"
    cache_dir: str = "./models"
    download_dir: str = "./models"
    trust_remote_code: bool = False


@dataclass
class EngineConfig:
    """vLLM 엔진 설정 클래스"""
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = None
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    swap_space: int = 4
    enforce_eager: bool = False
    disable_log_stats: bool = False
    dtype: str = "auto"
    sliding_window: Optional[int] = None


@dataclass
class QuantizationConfig:
    """양자화 설정 클래스"""
    enabled: bool = False
    method: Optional[str] = None
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True


@dataclass
class DistributedConfig:
    """분산 처리 설정 클래스"""
    world_size: int = 1
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"
    timeout: int = 1800


@dataclass
class TokenizerConfig:
    """토크나이저 설정 클래스"""
    trust_remote_code: bool = False
    padding_side: str = "left"
    truncation_side: str = "right"
    legacy: bool = False


@dataclass
class InferenceConfig:
    """추론 설정 클래스"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: Optional[int] = None


@dataclass
class MonitoringConfig:
    """모니터링 설정 클래스"""
    enabled: bool = True
    metrics_port: int = 8005
    log_stats_interval: int = 10
    profile_interval: int = 60
    prometheus: bool = True
    record_latency: bool = True
    record_memory: bool = True


@dataclass
class CachingConfig:
    """캐싱 설정 클래스"""
    enabled: bool = True
    prompt_cache_size: int = 1000
    result_cache_size: int = 1000
    ttl: int = 3600


@dataclass
class LoggingConfig:
    """로깅 설정 클래스"""
    level: str = "info"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    json: bool = False
    log_requests: bool = True
    log_responses: bool = False


@dataclass
class MAIVLLMConfig:
    """mai-vllm-serving 전체 설정 클래스"""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "server": self.server.__dict__,
            "model": self.model.__dict__,
            "engine": self.engine.__dict__,
            "quantization": self.quantization.__dict__,
            "distributed": self.distributed.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "inference": self.inference.__dict__,
            "monitoring": self.monitoring.__dict__,
            "caching": self.caching.__dict__,
            "logging": self.logging.__dict__
        }

    def to_json(self, indent: int = 2) -> str:
        """설정을 JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MAIVLLMConfig':
        """딕셔너리에서 설정 객체 생성"""
        server_config = ServerConfig(**config_dict.get("server", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        engine_config = EngineConfig(**config_dict.get("engine", {}))
        quantization_config = QuantizationConfig(**config_dict.get("quantization", {}))
        distributed_config = DistributedConfig(**config_dict.get("distributed", {}))
        tokenizer_config = TokenizerConfig(**config_dict.get("tokenizer", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))
        monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
        caching_config = CachingConfig(**config_dict.get("caching", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            server=server_config,
            model=model_config,
            engine=engine_config,
            quantization=quantization_config,
            distributed=distributed_config,
            tokenizer=tokenizer_config,
            inference=inference_config,
            monitoring=monitoring_config,
            caching=caching_config,
            logging=logging_config
        )


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않는 경우
        yaml.YAMLError: YAML 파싱 오류
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def load_config(config_path: Optional[str] = None) -> MAIVLLMConfig:
    """
    설정 파일 로드 및 MAIVLLMConfig 객체 생성

    Args:
        config_path: 설정 파일 경로 (None인 경우 기본 경로 사용)

    Returns:
        MAIVLLMConfig 객체
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # 설정 파일 로드
    config_dict = load_yaml_config(config_path)

    # 환경 변수로 설정 덮어쓰기
    config_dict = override_config_from_env(config_dict)

    # MAIVLLMConfig 객체 생성
    config = MAIVLLMConfig.from_dict(config_dict)

    return config


def override_config_from_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    환경 변수로 설정 덮어쓰기

    Args:
        config: 설정 딕셔너리

    Returns:
        환경 변수로 덮어쓴 설정 딕셔너리
    """
    # 설정 복사
    config_copy = copy.deepcopy(config)

    # 환경 변수 처리
    env_var_prefix = "VLLM_"

    # 서버 설정
    if "server" in config_copy:
        config_copy["server"]["host"] = os.environ.get(env_var_prefix + "HOST", config_copy["server"]["host"])
        config_copy["server"]["port"] = int(os.environ.get(env_var_prefix + "PORT", config_copy["server"]["port"]))
        config_copy["server"]["workers"] = int(
            os.environ.get(env_var_prefix + "WORKERS", config_copy["server"]["workers"]))
        config_copy["server"]["log_level"] = os.environ.get(env_var_prefix + "LOG_LEVEL",
                                                            config_copy["server"]["log_level"])

    # 모델 설정
    if "model" in config_copy:
        config_copy["model"]["name"] = os.environ.get(env_var_prefix + "MODEL", config_copy["model"]["name"])
        config_copy["model"]["revision"] = os.environ.get(env_var_prefix + "MODEL_REVISION",
                                                          config_copy["model"]["revision"])
        config_copy["model"]["cache_dir"] = os.environ.get(env_var_prefix + "MODEL_CACHE_DIR",
                                                           config_copy["model"]["cache_dir"])
        config_copy["model"]["trust_remote_code"] = os.environ.get(env_var_prefix + "TRUST_REMOTE_CODE", "0") == "1"

    # 엔진 설정
    if "engine" in config_copy:
        if env_var_prefix + "TENSOR_PARALLEL_SIZE" in os.environ:
            tensor_parallel_size = os.environ.get(env_var_prefix + "TENSOR_PARALLEL_SIZE")
            config_copy["engine"]["tensor_parallel_size"] = int(
                tensor_parallel_size) if tensor_parallel_size.lower() != "none" else None

        if env_var_prefix + "GPU_MEMORY_UTILIZATION" in os.environ:
            config_copy["engine"]["gpu_memory_utilization"] = float(
                os.environ.get(env_var_prefix + "GPU_MEMORY_UTILIZATION"))

        if env_var_prefix + "MAX_NUM_SEQS" in os.environ:
            config_copy["engine"]["max_num_seqs"] = int(os.environ.get(env_var_prefix + "MAX_NUM_SEQS"))

        if env_var_prefix + "MAX_NUM_BATCHED_TOKENS" in os.environ:
            config_copy["engine"]["max_num_batched_tokens"] = int(
                os.environ.get(env_var_prefix + "MAX_NUM_BATCHED_TOKENS"))

        if env_var_prefix + "SWAP_SPACE" in os.environ:
            config_copy["engine"]["swap_space"] = int(os.environ.get(env_var_prefix + "SWAP_SPACE"))

        config_copy["engine"]["dtype"] = os.environ.get(env_var_prefix + "DTYPE", config_copy["engine"]["dtype"])

    # 양자화 설정
    if "quantization" in config_copy:
        config_copy["quantization"]["enabled"] = os.environ.get(env_var_prefix + "QUANTIZATION_ENABLED", "0") == "1"
        config_copy["quantization"]["method"] = os.environ.get(env_var_prefix + "QUANTIZATION_METHOD",
                                                               config_copy["quantization"]["method"])

    # 분산 처리 설정
    if "distributed" in config_copy:
        if env_var_prefix + "WORLD_SIZE" in os.environ:
            config_copy["distributed"]["world_size"] = int(os.environ.get(env_var_prefix + "WORLD_SIZE"))

        config_copy["distributed"]["backend"] = os.environ.get(env_var_prefix + "BACKEND",
                                                               config_copy["distributed"]["backend"])
        config_copy["distributed"]["master_addr"] = os.environ.get("MASTER_ADDR",
                                                                   config_copy["distributed"]["master_addr"])
        config_copy["distributed"]["master_port"] = os.environ.get("MASTER_PORT",
                                                                   config_copy["distributed"]["master_port"])

    return config_copy


def get_config_from_args() -> MAIVLLMConfig:
    """
    명령행 인수에서 설정 파일 경로를 추출하고 설정 로드

    Returns:
        MAIVLLMConfig 객체
    """
    parser = argparse.ArgumentParser(description="mai-vllm-serving 설정")
    parser.add_argument("--config", type=str, default=None, help="설정 파일 경로")
    args, _ = parser.parse_known_args()

    return load_config(args.config)


# 모듈 레벨 설정 인스턴스 (싱글톤)
_config_instance = None


def get_config() -> MAIVLLMConfig:
    """
    전역 설정 인스턴스 가져오기 (싱글톤 패턴)

    Returns:
        MAIVLLMConfig 객체
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = get_config_from_args()

    return _config_instance


def reset_config():
    """전역 설정 인스턴스 리셋"""
    global _config_instance
    _config_instance = None


# 테스트용 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 설정 로드
    config = get_config()

    # 설정 출력
    print("===== Mai-VLLM-Serving Configuration =====")
    print(config.to_json())
