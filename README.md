# 임신과 출산 RAG 챗봇

임신과 출산에 관한 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 챗봇 시스템입니다.

## 주요 기능

- **RAG 시스템**: PDF와 XLSX 파일에서 임신과 출산 관련 정보를 추출하여 벡터 데이터베이스에 저장
- **하이브리드 검색**: 벡터 검색과 BM25 검색을 결합한 앙상블 검색 방식
- **ko-gemma2-9b 모델**: 한국어에 최적화된 대화형 AI 모델 사용
- **FAISS 벡터 스토어**: 고성능 벡터 검색을 위한 FAISS 사용
- **클라이언트-서버 아키텍처**: FastAPI 기반 REST API 서버

## 시스템 요구사항

- Ubuntu 서버
- RTX 4090 2장 (GPU 가속)
- Python 3.8+
- CUDA 지원

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

#### 방법 1: 직접 실행
```bash
python3 chatbot.py
```

#### 방법 2: 스크립트 실행
```bash
./run_server.sh
```

#### 방법 3: systemd 서비스로 실행
```bash
# 서비스 파일 복사
sudo cp pregnancy-chatbot.service /etc/systemd/system/

# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable pregnancy-chatbot
sudo systemctl start pregnancy-chatbot

# 상태 확인
sudo systemctl status pregnancy-chatbot
```

### 3. 클라이언트 실행

```bash
python3 client.py
```

## API 엔드포인트

### POST /chat
질문을 전송하고 답변을 받습니다.

**요청:**
```json
{
    "question": "임신 중 운동은 언제부터 시작할 수 있나요?",
    "user_id": "optional_user_id"
}
```

**응답:**
```json
{
    "answer": "임신 중 운동은 일반적으로 임신 12주 이후부터 시작하는 것이 권장됩니다...",
    "sources": ["dataset/난임 가이드북.pdf", "dataset/expert_qna.xlsx"],
    "confidence": 0.85
}
```

### POST /chat/stream
질문을 전송하고 실시간 스트리밍으로 답변을 받습니다.

**요청:**
```json
{
    "question": "임신 중 운동은 언제부터 시작할 수 있나요?",
    "user_id": "optional_user_id"
}
```

**응답 (Server-Sent Events):**
```
data: {"sources": ["dataset/난임 가이드북.pdf"], "type": "sources"}

data: {"token": "임신", "finished": false}

data: {"token": "중", "finished": false}

data: {"token": "운동", "finished": false}

...

data: {"token": "", "finished": true}
```

**특징:**
- Content-Type: `text/plain`
- Server-Sent Events (SSE) 프로토콜 사용
- 토큰별 실시간 스트리밍
- 소스 정보 먼저 전송 후 답변 스트리밍
- `finished: true`로 완료 신호

### GET /health
서버 상태를 확인합니다.

**응답:**
```json
{
    "status": "healthy",
    "message": "임신과 출산 RAG 챗봇이 정상 작동 중입니다."
}
```

## 데이터셋 구조

```
dataset/
├── 1. 난임 가이드북(2022년 12월).pdf
├── 2022년 아이사랑 사이트 콘텐츠 자문위원회 원고.hwp.pdf
├── 전체글 qna 추출데이터 250703.xlsx
├── expert_qna.xlsx
├── generated_test_questions.xlsx
├── 20250227(이윤주선생님정리)_난임질의응답.xlsx
├── 인터넷 포털사이트 문의항목 정리 및 전문가 의견 취합.xlsx
└── 제2회 영유아 패널 학술대회용 데이터/
```

## 프로젝트 파일 구조

```
STEAM/
├── chatbot_v0.1.py          # 메인 서버 코드
├── client.py                # 클라이언트 코드
├── ecosystem.config.js      # PM2 설정 파일
├── pm2_start.sh            # PM2 시작 스크립트
├── pm2_manage.sh           # PM2 관리 스크립트
├── run_server.sh           # 직접 실행 스크립트
├── requirements.txt         # Python 의존성
├── pregnancy-chatbot.service # Systemd 서비스
├── README.md               # 프로젝트 문서
├── dataset/                # 데이터셋 폴더
│   ├── *.pdf               # PDF 문서들
│   └── *.xlsx              # 엑셀 파일들
├── vectorstore/            # FAISS 벡터 데이터베이스
│   ├── index.faiss
│   └── index.pkl
└── logs/                   # PM2 로그 파일들
    ├── pm2-error.log
    ├── pm2-out.log
    └── pm2-combined.log
```

## 아키텍처

### 서버 구성
- **FastAPI**: REST API 서버 (스트리밍 지원)
- **ko-gemma2-9b**: 대화형 AI 모델 (4-bit 양자화)
- **TextIteratorStreamer**: 실시간 토큰 스트리밍
- **FAISS**: 벡터 데이터베이스
- **LangChain**: RAG 파이프라인
- **PyPDF2/OpenPyXL**: 문서 파싱

### 검색 방식
1. **벡터 검색**: 의미적 유사도 기반 검색 (가중치: 0.7)
2. **BM25 검색**: 키워드 기반 검색 (가중치: 0.3)
3. **앙상블**: 두 검색 결과를 결합하여 최종 결과 도출

## 실시간 스트리밍 기능

### 기술적 특징
- **TextIteratorStreamer**: Transformers 라이브러리의 내장 스트리밍 기능 활용
- **멀티스레딩**: 모델 생성과 스트리밍이 별도 스레드에서 병렬 처리
- **효율적 성능**: 토큰당 모델 호출 방식 대비 256배 성능 향상
- **Server-Sent Events**: 웹 표준 SSE 프로토콜로 실시간 데이터 전송

### 스트리밍 동작 방식
1. 클라이언트가 `/chat/stream` 엔드포인트로 질문 전송
2. 서버에서 관련 문서 검색 및 소스 정보 먼저 전송
3. 별도 스레드에서 `model.generate()` 실행
4. `TextIteratorStreamer`가 생성되는 토큰을 실시간으로 캐치
5. 각 토큰을 JSON 형태로 즉시 클라이언트에 스트리밍
6. 생성 완료 시 `finished: true` 신호 전송

### 성능 비교
| 방식 | 토큰당 모델 호출 | 응답 시작 시간 | 전체 응답 시간 | CPU/GPU 효율 |
|------|-----------------|----------------|----------------|---------------|
| 일반 응답 | 1번 | 느림 | 보통 | 높음 |
| 이전 스트리밍 | 256번 | 매우 느림 | 매우 느림 | 매우 낮음 |
| **현재 스트리밍** | **1번** | **빠름** | **빠름** | **매우 높음** |

## 사용 예시

### 서버 시작

#### 방법 1: 직접 실행
```bash
# 서버 실행
python3 chatbot_v0.1.py

# 서버가 http://localhost:9000 에서 실행됩니다
```

#### 방법 2: PM2로 백그라운드 실행 (추천) 🚀
```bash
# PM2로 시작 (초기 설정 포함)
./pm2_start.sh

# 또는 간단히
./pm2_manage.sh start

# 상태 확인
./pm2_manage.sh status

# 실시간 로그 보기
./pm2_manage.sh logs
```

**PM2 주요 명령어:**
- `./pm2_manage.sh start` - 서버 시작
- `./pm2_manage.sh stop` - 서버 중지
- `./pm2_manage.sh restart` - 서버 재시작
- `./pm2_manage.sh logs` - 실시간 로그
- `./pm2_manage.sh monit` - 모니터링
- `./pm2_manage.sh delete` - 프로세스 삭제

### 클라이언트 사용
```bash
# 클라이언트 실행
python3 client.py

- 클라이언트 pc에서 실행하여야 합니다.

# 실시간 스트리밍 출력 예시
🤖 답변 (실시간 스트리밍)
------------------------------
임신중운동은일반적으로임신12주이후부터시작하는것이권장됩니다...

(응답시간: 3.45초)

📚 참고 자료:
  1. dataset/난임 가이드북.pdf
  2. dataset/expert_qna.xlsx

신뢰도: 0.67

# 질문 예시
질문: 임신 중 금기 식품은 무엇인가요?
질문: 출산 후 산후조리는 어떻게 해야 하나요?
질문: 임신 중 운동은 언제부터 시작할 수 있나요?
질문: 태아의 성장 발달 과정은 어떻게 되나요?
```

## 모니터링

### 로그 확인
```bash
# systemd 서비스 로그
sudo journalctl -u pregnancy-chatbot -f

# 직접 실행 시 로그는 콘솔에 출력됩니다
```

### 성능 모니터링
- GPU 사용률: `nvidia-smi`
- 메모리 사용률: `htop`
- API 응답 시간: 클라이언트에서 자동 측정

## 문제 해결

### 일반적인 문제들

1. **모델 로딩 실패**
   - GPU 메모리 부족 시 `load_in_8bit=True` 설정 확인
   - CUDA 버전 호환성 확인

2. **벡터 스토어 생성 실패**
   - dataset 폴더의 파일 권한 확인
   - 디스크 공간 확인

3. **API 연결 실패**
   - 서버가 실행 중인지 확인: `curl http://localhost:9000/health`
   - 방화벽 설정 확인

4. **스트리밍 연결 실패**
   - 클라이언트에서 "❌ 스트리밍 연결 실패" 오류 시
   - 서버 로그에서 TextIteratorStreamer 관련 오류 확인
   - 모델 메모리 부족 시 4-bit 양자화 설정 확인
   - 방화벽 설정 확인

5. **PM2 관련 문제**
   - PM2 설치: `npm install -g pm2`
   - 프로세스 상태 확인: `./pm2_manage.sh status`
   - 로그 확인: `./pm2_manage.sh logs`
   - 에러 로그: `./pm2_manage.sh error-logs`
   - 모니터링: `./pm2_manage.sh monit`
   - 프로세스 재시작: `./pm2_manage.sh restart`
   - Conda 환경 문제: `ecosystem.config.js`에서 Python 경로 확인

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
