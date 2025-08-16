#!/bin/bash

echo "Python 3.11용 패키지 설치 스크립트"
echo "=================================="

# Python 버전 확인
echo "Python 버전 확인 중..."
python3 --version

# pip 업그레이드
echo "pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# 기본 패키지들 먼저 설치
echo "기본 패키지 설치 중..."
python3 -m pip install numpy==1.24.3
python3 -m pip install pandas==2.1.4
python3 -m pip install requests==2.31.0

# FastAPI 관련 패키지 설치
echo "FastAPI 관련 패키지 설치 중..."
python3 -m pip install fastapi==0.104.1
python3 -m pip install uvicorn==0.24.0
python3 -m pip install pydantic==2.5.0
python3 -m pip install python-multipart==0.0.6

# 문서 처리 패키지 설치
echo "문서 처리 패키지 설치 중..."
python3 -m pip install pypdf2==3.0.1
python3 -m pip install openpyxl==3.1.2

# LangChain 패키지 설치
echo "LangChain 패키지 설치 중..."
python3 -m pip install langchain==0.1.0
python3 -m pip install langchain-community==0.0.10
python3 -m pip install langchain-text-splitters==0.0.1

# 벡터 스토어 및 임베딩 패키지 설치
echo "벡터 스토어 패키지 설치 중..."
python3 -m pip install faiss-cpu==1.7.4
python3 -m pip install sentence-transformers==2.2.2

# AI 모델 관련 패키지 설치
echo "AI 모델 패키지 설치 중..."
python3 -m pip install transformers==4.36.2
python3 -m pip install accelerate==0.25.0
python3 -m pip install tiktoken==0.5.2

# PyTorch 설치 (CUDA 지원)
echo "PyTorch 설치 중..."
python3 -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# bitsandbytes 설치 (GPU 양자화)
echo "bitsandbytes 설치 중..."
python3 -m pip install bitsandbytes==0.41.3

echo "=================================="
echo "모든 패키지 설치 완료!"
echo "설치된 패키지 확인:"
python3 -m pip list | grep -E "(fastapi|uvicorn|langchain|faiss|transformers|torch)"
