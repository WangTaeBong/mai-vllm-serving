#!/bin/bash

# 컨테이너 설정
IMAGE_NAME="mai-vllm-serving:latest"
CONTAINER_NAME="mai-vllm-serving"

# 모델 및 로그 디렉토리 설정 (필요에 따라 수정)
MODEL_DIR="/data/vllm-models"
LOG_DIR="/applog/mai-vllm-serving"
CONFIG_DIR="/app/mai-vllm-serving/configs"

# 포트 설정 (기본값: 8020)
PORT=8020

# 사용자 설정 - 원하는 사용자로 실행 (비어있으면 컨테이너 기본값 사용)
use_user='aicess' # 예: use_user='aicess'

# 사용자 ID 설정
docker_user=''
if [ "$use_user" != '' ]; then
    user_uid=$(id -u $use_user)
    user_gid=$(id -g $use_user)
    docker_user="--user $user_uid:$user_gid"
    echo "Docker를 사용자 $use_user (UID:$user_uid, GID:$user_gid)로 실행합니다."
else
    echo "Docker를 컨테이너 기본 사용자로 실행합니다."
fi

# 컨테이너 상태 확인 함수
check_container_status() {
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "실행 중"
        return 0
    elif [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        echo "정지됨"
        return 1
    else
        echo "없음"
        return 2
    fi
}

# 컨테이너 상태 상세 정보 출력 함수
show_container_info() {
    echo "===== 컨테이너 상태 정보 ====="
    status=$(check_container_status)
    echo "현재 상태: ${status}"

    if [ "${status}" != "없음" ]; then
        echo -e "\n컨테이너 상세 정보:"
        docker ps -a -f name=${CONTAINER_NAME} --format "ID:\t{{.ID}}\n이름:\t{{.Names}}\n이미지:\t{{.Image}}\n상태:\t{{.Status}}\n포트:\t{{.Ports}}\n생성됨:\t{{.CreatedAt}}"

        if [ "${status}" == "실행 중" ]; then
            echo -e "\n컨테이너 리소스 사용량:"
            docker stats ${CONTAINER_NAME} --no-stream
        fi
    fi
}

# 컨테이너 시작 함수
start_container() {
    echo "===== mai-vllm-serving 컨테이너 시작 중 ====="
    status=$(check_container_status)

    if [ "${status}" == "실행 중" ]; then
        echo "컨테이너가 이미 실행 중입니다."
        return 0
    elif [ "${status}" == "정지됨" ]; then
        echo "정지된 컨테이너를 시작합니다..."
        docker start ${CONTAINER_NAME}
    else
        echo "새 컨테이너를 생성하고 시작합니다..."

        # 필요한 디렉토리 생성
        mkdir -p ${LOG_DIR}

        # 로그 파일 경로 설정 - 컨테이너 내부의 로그 디렉토리에 저장
        INTERNAL_LOG_DIR="/app/internal-logs"
        LOG_FILE_PATH="${INTERNAL_LOG_DIR}/mai-vllm-serving.log"

        # 컨테이너 실행
        docker run --gpus all \
            --name ${CONTAINER_NAME} \
            ${docker_user} \
            -v ${MODEL_DIR}:/app/models \
            -v ${LOG_DIR}:/app/logs \
            -v ${CONFIG_DIR}:/app/configs \
            -p ${PORT}:8020 \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e LOGGING_FILE_PATH="${LOG_FILE_PATH}" \
            -e HF_HOME="/app/.cache/huggingface" \
            -e TRANSFORMERS_CACHE="/app/.cache/huggingface" \
            -e XDG_CACHE_HOME="/app/.cache" \
            --entrypoint=/entrypoint.sh \
            -d \
            ${IMAGE_NAME} \
            python3 -m mai_vllm_serving.server --log-file ${LOG_FILE_PATH}
    fi

    # 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo "===== 컨테이너 시작 성공! ====="
        echo "서비스 접속: http://localhost:${PORT}"

        # 컨테이너 로그 출력
        echo -e "\n처음 10줄의 로그 출력:"
        sleep 2
        docker logs ${CONTAINER_NAME} 2>&1 | head -10
    else
        echo "===== 컨테이너 시작 실패! ====="
        return 1
    fi
}

# 컨테이너 중지 함수
stop_container() {
    echo "===== mai-vllm-serving 컨테이너 중지 중 ====="
    status=$(check_container_status)

    if [ "${status}" == "실행 중" ]; then
        echo "컨테이너를 중지합니다..."
        docker stop ${CONTAINER_NAME}

        if [ $? -eq 0 ]; then
            echo "===== 컨테이너 중지 성공! ====="
        else
            echo "===== 컨테이너 중지 실패! ====="
            return 1
        fi
    elif [ "${status}" == "정지됨" ]; then
        echo "컨테이너가 이미 정지되어 있습니다."
    else
        echo "실행 중인 컨테이너가 없습니다."
    fi
}

# 컨테이너 재시작 함수
restart_container() {
    echo "===== mai-vllm-serving 컨테이너 재시작 중 ====="
    status=$(check_container_status)

    if [ "${status}" == "실행 중" ] || [ "${status}" == "정지됨" ]; then
        echo "컨테이너를 중지합니다..."
        docker stop ${CONTAINER_NAME} >/dev/null 2>&1

        echo "컨테이너를 시작합니다..."
        docker start ${CONTAINER_NAME}

        if [ $? -eq 0 ]; then
            echo "===== 컨테이너 재시작 성공! ====="
            echo "서비스 접속: http://localhost:${PORT}"
        else
            echo "===== 컨테이너 재시작 실패! ====="
            return 1
        fi
    else
        echo "재시작할 컨테이너가 없습니다. 새로 시작합니다..."
        start_container
    fi
}

# 컨테이너 로그 출력 함수
show_logs() {
    echo "===== mai-vllm-serving 컨테이너 로그 ====="
    status=$(check_container_status)

    if [ "${status}" == "실행 중" ] || [ "${status}" == "정지됨" ]; then
        if [ -z "$1" ]; then
            # 기본: 마지막 100줄
            docker logs --tail 100 ${CONTAINER_NAME}
        elif [ "$1" == "follow" ]; then
            # 실시간 로그 확인
            docker logs --tail 20 -f ${CONTAINER_NAME}
        else
            # 특정 줄 수 출력
            docker logs --tail $1 ${CONTAINER_NAME}
        fi
    else
        echo "컨테이너가 존재하지 않습니다."
        return 1
    fi
}

# 컨테이너 완전 제거 함수
remove_container() {
    echo "===== mai-vllm-serving 컨테이너 제거 중 ====="
    status=$(check_container_status)

    if [ "${status}" == "실행 중" ]; then
        echo "실행 중인 컨테이너를 중지합니다..."
        docker stop ${CONTAINER_NAME}
    fi

    if [ "${status}" != "없음" ]; then
        echo "컨테이너를 제거합니다..."
        docker rm ${CONTAINER_NAME}

        if [ $? -eq 0 ]; then
            echo "===== 컨테이너 제거 성공! ====="
        else
            echo "===== 컨테이너 제거 실패! ====="
            return 1
        fi
    else
        echo "제거할 컨테이너가 없습니다."
    fi
}

# 도움말 출력 함수
show_help() {
    echo "사용법: $0 [명령]"
    echo ""
    echo "명령:"
    echo "  start      - 컨테이너 시작 또는 생성"
    echo "  stop       - 컨테이너 중지"
    echo "  restart    - 컨테이너 재시작"
    echo "  status     - 컨테이너 상태 확인"
    echo "  logs [N]   - 컨테이너 로그 확인 (기본: 마지막 100줄, N: 출력할 줄 수)"
    echo "  logs follow - 실시간 로그 확인"
    echo "  remove     - 컨테이너 제거"
    echo "  help       - 도움말 표시"
    echo ""
    echo "사용자 설정:"
    echo "  스크립트 상단의 use_user 변수를 편집하여 실행 사용자를 변경할 수 있습니다."
    echo "  use_user='username'  # 특정 사용자로 실행"
    echo "  use_user=''          # 컨테이너 기본 사용자로 실행"
    echo ""
    echo "예시:"
    echo "  $0 start    # 컨테이너 시작"
    echo "  $0 logs 50  # 마지막 50줄의 로그 출력"
    echo "  $0 status   # 컨테이너 상태 확인"
}

# 메인 스크립트 로직
case "$1" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    status)
        show_container_info
        ;;
    logs)
        show_logs "$2"
        ;;
    remove)
        remove_container
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "알 수 없는 명령: $1"
        show_help
        exit 1
        ;;
esac

exit 0