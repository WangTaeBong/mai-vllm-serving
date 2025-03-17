FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# NVIDIA 엔트리포인트 스크립트 비활성화 설정
ENV NVIDIA_DISABLE_REQUIRE=true
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# 기본 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 기존 엔트리포인트 재정의
ENTRYPOINT []
COPY entrypoint-reset.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 비루트 사용자 생성
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user:user /app

# Python 패키지 설치
COPY --chown=user:user requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# 애플리케이션 코드 복사
COPY --chown=user:user . /app/

# 설정 파일 확인 스크립트 추가
COPY --chown=user:user config-path-check.sh /app/
RUN chmod +x /app/config-path-check.sh
RUN /app/config-path-check.sh

# 필요한 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/logs /app/profiles /app/models /app/internal-logs /app/.cache && \
    chmod -R 777 /app/logs /app/profiles /app/models /app/internal-logs /app/.cache && \
    chown -R user:user /app/logs /app/profiles /app/models /app/internal-logs /app/.cache

# vLLM을 위한 환경 변수 설정
ENV OMP_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

# 비루트 사용자로 전환
USER user

# 기본 명령 없음 - 별도 스크립트에서 실행될 예정