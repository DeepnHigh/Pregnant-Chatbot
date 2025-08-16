#!/bin/bash

echo "4-bit 양자화 패키지 설치 스크립트"
echo "=================================="

# Anaconda 환경 활성화
conda activate preg

# 4-bit 양자화를 위한 패키지 설치
echo "4-bit 양자화 패키지 설치 중..."
pip install optimum==1.20.1
pip install auto-gptq==0.8.0

# 기존 패키지들도 업데이트
echo "기존 패키지 업데이트 중..."
pip install --upgrade transformers
pip install --upgrade accelerate
pip install --upgrade bitsandbytes

echo "설치 완료!"
echo "이제 gemma-9b 모델을 4-bit 양자화로 실행할 수 있습니다."
