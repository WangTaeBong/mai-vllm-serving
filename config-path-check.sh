#!/bin/bash

# 컨테이너에서 사용할 수 있는 설정 파일 확인 스크립트
# 이 스크립트는 Dockerfile에 추가하여 빌드 과정에서 실행할 수 있습니다.

echo "===== 설정 파일 경로 확인 ====="
echo "현재 디렉토리: $(pwd)"

# configs 디렉토리 확인
if [ -d "./configs" ]; then
    echo "configs 디렉토리가 존재합니다."
    ls -la ./configs
    
    # YAML 파일 확인
    if [ -f "./configs/default_config.yaml" ]; then
        echo "default_config.yaml 파일을 찾았습니다."
    else
        echo "경고: default_config.yaml 파일을 찾을 수 없습니다."
    fi
else
    echo "경고: configs 디렉토리가 존재하지 않습니다."
fi

# 로그 디렉토리 권한 확인
if [ -d "./logs" ]; then
    echo "logs 디렉토리 권한:"
    ls -la ./logs
else
    echo "logs 디렉토리가 아직 생성되지 않았습니다."
fi

# Python 경로 확인
which python || echo "python 명령을 찾을 수 없습니다."
which python3 || echo "python3 명령을 찾을 수 없습니다."

# Python 버전 확인
python3 --version || echo "Python 버전을 확인할 수 없습니다."

echo "===== 확인 완료 ====="