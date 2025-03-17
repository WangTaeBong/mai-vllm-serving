FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN mkdir -p $WORKDIR/cache
RUN chmod 777 $WORKDIR/cache

# 기본 패키지 업데이트 및 Python 3.11 설치
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
    software-properties-common \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt update && apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 기본 Python을 Python 3.11로 변경
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# pip 최신 버전으로 업그레이드
RUN python3 -m pip install --upgrade pip

# 타임존 설정 - Asia/Seoul
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    echo "Asia/Seoul" > /etc/timezone

# 타임존 환경변수 설정
ENV TZ=Asia/Seoul

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app/

# 설정 파일 확인 스크립트 추가
COPY config-path-check.sh /app/
RUN chmod +x /app/config-path-check.sh
RUN /app/config-path-check.sh

# 필요한 디렉토리 생성 및 권한 설정 - 777 권한으로 설정하여 누구나 접근 가능하게 함
RUN mkdir -p /app/logs /app/profiles /app/models /app/internal-logs /app/cache/huggingface && \
    chmod -R 777 /app/logs /app/profiles /app/models /app/internal-logs /app/cache

# vLLM을 위한 환경 변수 설정
ENV OMP_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HF_HOME=/app/cache/huggingface \
    XDG_CACHE_HOME=/app/cache