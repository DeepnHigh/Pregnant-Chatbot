#!/bin/bash

# PM2로 임신 챗봇 시작하는 스크립트
# Conda 가상환경 preg에서 실행

echo "🚀 PM2로 임신 챗봇 서버를 시작합니다..."

# 로그 디렉토리 생성
mkdir -p logs

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate preg

# PM2가 설치되어 있지 않으면 설치
if ! command -v pm2 &> /dev/null; then
    echo "📦 PM2를 설치합니다..."
    npm install -g pm2
fi

# 기존 프로세스가 있다면 중지
pm2 stop pregnancy-chatbot 2>/dev/null || true
pm2 delete pregnancy-chatbot 2>/dev/null || true

# PM2로 애플리케이션 시작
echo "🔧 ecosystem.config.js로 애플리케이션을 시작합니다..."
pm2 start ecosystem.config.js

# 상태 확인
echo "📊 PM2 프로세스 상태:"
pm2 status

# 로그 확인 방법 안내
echo ""
echo "📝 로그 확인 방법:"
echo "  실시간 로그: pm2 logs pregnancy-chatbot"
echo "  에러 로그:   pm2 logs pregnancy-chatbot --err"
echo "  출력 로그:   pm2 logs pregnancy-chatbot --out"
echo ""
echo "🎯 관리 명령어:"
echo "  재시작:     pm2 restart pregnancy-chatbot"
echo "  중지:       pm2 stop pregnancy-chatbot"
echo "  삭제:       pm2 delete pregnancy-chatbot"
echo "  모니터링:   pm2 monit"
echo ""
echo "✅ 챗봇 서버가 PM2로 시작되었습니다! (포트: 9000)"
