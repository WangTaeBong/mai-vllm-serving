#!/bin/bash

# 스크립트 시작 메시지
echo "===== mai-vllm-serving Docker 이미지 빌드 시작 ====="
echo "시작 시간: $(date)"

# 엔트리포인트 스크립트 생성
cat > entrypoint-reset.sh << 'EOF'
#!/bin/bash
# 이 스크립트는 NVIDIA 컨테이너의 기본 엔트리포인트를 재정의하기 위한 것입니다

# 전달된 모든 인수를 그대로 실행합니다
exec "$@"
EOF

chmod +x entrypoint-reset.sh
echo "엔트리포인트 재정의 스크립트 생성 완료"

# 빌드할 이미지 이름과 태그 설정
IMAGE_NAME="mai-vllm-serving"
TAG="latest"

# 전체 이미지 이름
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# 빌드 시작 전 정보 출력
echo "빌드할 이미지: ${FULL_IMAGE_NAME}"
echo "빌드 컨텍스트: $(pwd)"

# 빌드 시작
echo "Docker 빌드 시작..."
docker build -t ${FULL_IMAGE_NAME} . --no-cache

# 빌드 결과 확인
if [ $? -eq 0 ]; then
    echo "===== 빌드 성공! ====="
    echo "빌드된 이미지 정보:"
    docker images ${IMAGE_NAME}

    echo -e "\n이미지 실행 방법 예시:"
    echo "docker run --gpus all -it --entrypoint=\"\" ${FULL_IMAGE_NAME} python3 -m mai_vllm_serving.server"

    # 빌드 후 임시 파일 정리
    rm -f entrypoint-reset.sh
else
    echo "===== 빌드 실패! ====="
    exit 1
fi

echo "완료 시간: $(date)"