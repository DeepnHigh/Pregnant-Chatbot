#!/bin/bash

# PM2 임신 챗봇 관리 스크립트

APP_NAME="pregnancy-chatbot"

case "$1" in
    start)
        echo "🚀 챗봇 시작..."
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate preg
        pm2 start ecosystem.config.js
        pm2 status
        ;;
    stop)
        echo "🛑 챗봇 중지..."
        pm2 stop $APP_NAME
        pm2 status
        ;;
    restart)
        echo "🔄 챗봇 재시작..."
        pm2 restart $APP_NAME
        pm2 status
        ;;
    status)
        echo "📊 챗봇 상태:"
        pm2 status $APP_NAME
        ;;
    logs)
        echo "📝 실시간 로그 (Ctrl+C로 종료):"
        pm2 logs $APP_NAME
        ;;
    error-logs)
        echo "❌ 에러 로그:"
        pm2 logs $APP_NAME --err --lines 50
        ;;
    out-logs)
        echo "✅ 출력 로그:"
        pm2 logs $APP_NAME --out --lines 50
        ;;
    monit)
        echo "📈 PM2 모니터링 (q로 종료):"
        pm2 monit
        ;;
    delete)
        echo "🗑️  챗봇 프로세스 삭제..."
        pm2 stop $APP_NAME
        pm2 delete $APP_NAME
        echo "✅ 삭제 완료"
        ;;
    reload)
        echo "🔄 무중단 재시작..."
        pm2 reload $APP_NAME
        pm2 status
        ;;
    flush)
        echo "🧹 로그 파일 정리..."
        pm2 flush $APP_NAME
        echo "✅ 로그 정리 완료"
        ;;
    save)
        echo "💾 현재 PM2 프로세스 목록 저장..."
        pm2 save
        echo "✅ 저장 완료"
        ;;
    startup)
        echo "🔧 시스템 부팅시 자동 시작 설정..."
        pm2 startup
        echo "위 명령어를 복사해서 실행하세요."
        ;;
    *)
        echo "🤖 PM2 임신 챗봇 관리 스크립트"
        echo ""
        echo "사용법: $0 {command}"
        echo ""
        echo "📋 사용 가능한 명령어:"
        echo "  start        - 챗봇 시작"
        echo "  stop         - 챗봇 중지"
        echo "  restart      - 챗봇 재시작"
        echo "  reload       - 무중단 재시작"
        echo "  status       - 상태 확인"
        echo "  logs         - 실시간 로그"
        echo "  error-logs   - 에러 로그"
        echo "  out-logs     - 출력 로그"
        echo "  monit        - 모니터링"
        echo "  delete       - 프로세스 삭제"
        echo "  flush        - 로그 정리"
        echo "  save         - 프로세스 목록 저장"
        echo "  startup      - 부팅시 자동 시작"
        echo ""
        echo "📝 예시:"
        echo "  $0 start     # 챗봇 시작"
        echo "  $0 logs      # 실시간 로그 확인"
        echo "  $0 restart   # 재시작"
        echo ""
        exit 1
        ;;
esac
