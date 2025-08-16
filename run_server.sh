#!/bin/bash

# 임신과 출산 RAG 챗봇 서버 실행 스크립트 (Anaconda 환경용)

echo "임신과 출산 RAG 챗봇 서버를 시작합니다..."

# Anaconda preg 환경 활성화
source ~/.bashrc
conda activate preg

# 현재 환경 확인
echo "현재 Python 환경:"
python --version
echo "현재 conda 환경: $CONDA_DEFAULT_ENV"

# 서버 실행
echo "서버를 시작합니다..."
python chatbot.py
