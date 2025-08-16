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

## 아키텍처

### 서버 구성
- **FastAPI**: REST API 서버
- **ko-gemma2-9b**: 대화형 AI 모델
- **FAISS**: 벡터 데이터베이스
- **LangChain**: RAG 파이프라인
- **PyPDF2/OpenPyXL**: 문서 파싱

### 검색 방식
1. **벡터 검색**: 의미적 유사도 기반 검색 (가중치: 0.7)
2. **BM25 검색**: 키워드 기반 검색 (가중치: 0.3)
3. **앙상블**: 두 검색 결과를 결합하여 최종 결과 도출

## 사용 예시

### 서버 시작
```bash
# 서버 실행
python3 chatbot.py

# 서버가 http://localhost:8000 에서 실행됩니다
```

### 클라이언트 사용
```bash
# 클라이언트 실행
python3 client.py

# 질문 예시
질문: 임신 중 금기 식품은 무엇인가요?
질문: 출산 후 산후조리는 어떻게 해야 하나요?
질문: 임신 중 운동은 언제부터 시작할 수 있나요?
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
   - 서버가 실행 중인지 확인: `curl http://localhost:8000/health`
   - 방화벽 설정 확인

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
