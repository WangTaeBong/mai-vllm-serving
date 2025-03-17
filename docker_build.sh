#!/bin/bash

# 스크립트 시작 메시지
echo "===== mai-vllm-serving Docker 이미지 빌드 시작 ====="
echo "시작 시간: $(date)"

# 빌드할 이미지 이름과 태그 설정
IMAGE_NAME="mai-vllm-serving"
TAG="latest"

# 전체 이미지 이름
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# 빌드 시작 전 정보 출력
echo "빌드할 이미지: ${FULL_IMAGE_NAME}"
echo "빌드 컨텍스트: $(pwd)"

# 시스템 정보 출력
echo "===== 시스템 정보 ====="
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU 정보:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "경고: NVIDIA GPU가 감지되지 않았습니다."
fi

echo "CPU 정보:"
cat /proc/cpuinfo | grep "model name" | head -n 1

echo "메모리 정보:"
free -h | grep Mem

# 도커 정보 출력
echo "Docker 버전:"
docker --version

# 빌드 시작
echo -e "\n===== Docker 빌드 시작 ====="
# --no-cache 옵션 사용 여부를 사용자에게 묻기
read -p "클린 빌드를 수행할까요? (캐시 사용 안 함) [y/N]: " clean_build
if [[ $clean_build =~ ^[Yy]$ ]]; then
    CACHE_OPTION="--no-cache"
    echo "캐시를 사용하지 않는 클린 빌드를 수행합니다."
else
    CACHE_OPTION=""
    echo "캐시를 사용하여 빌드를 수행합니다."
fi

# 빌드 성능 향상을 위한 옵션 추가
docker build -t ${FULL_IMAGE_NAME} . ${CACHE_OPTION} --progress=plain

# 빌드 결과 확인
if [ $? -eq 0 ]; then
    echo "===== 빌드 성공! ====="
    echo "빌드된 이미지 정보:"
    docker images ${IMAGE_NAME} --format "이미지 ID: {{.ID}}\n이름: {{.Repository}}:{{.Tag}}\n크기: {{.Size}}\n생성일: {{.CreatedAt}}"

    echo -e "\n이미지 실행 방법 예시:"
    echo "docker run --gpus all -it ${FULL_IMAGE_NAME} python -m mai_vllm_serving.server"
    echo -e "\n또는 docker_run.sh 스크립트를 사용하세요:"
    echo "./docker_run.sh start"
else
    echo "===== 빌드 실패! ====="
    exit 1
fi

echo "완료 시간: $(date)"