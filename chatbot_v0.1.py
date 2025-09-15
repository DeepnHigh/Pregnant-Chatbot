import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
from transformers import TextIteratorStreamer

import threading
INITIAL_PROMPT = """
당신은 매우 유능한 임신과 출산 전문가인 산부인과 의사입니다. 질문에 대해 친절하게, 전문적으로 답변하세요. 임신과 출산에 관련된 질문이 아니면 '임신과 출산에 관련된 질문이 아닙니다.'라고 출력하고 답변하지 마세요.
"""

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class SimplePregnancyChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.documents = []
        
        # GPU 메모리 최적화 설정
        self.setup_gpu_memory()
    
    def setup_gpu_memory(self):
        """GPU 메모리 최적화 설정"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU 개수: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 메모리 캐시 정리
            torch.cuda.empty_cache()
            
            # 메모리 할당 전략 설정
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("GPU 메모리 최적화 설정 완료")
    
    def load_model(self):
        """gemma-9b 모델 로드 - 최적화 버전"""
        logger.info("gemma-9b 모델 로딩 중...")
        
        try:
            # gemma-9b 모델 사용
            model_name = "rtzr/ko-gemma-2-9b-it"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # GPU 개수 확인
            gpu_count = torch.cuda.device_count()
            logger.info(f"사용 가능한 GPU 개수: {gpu_count}")
            
            if gpu_count >= 2:
                # GPU 2개 이상인 경우 분산 배치
                logger.info("GPU 2개에 분산 배치 설정")
                device_map = "auto"
                # 더 보수적인 메모리 설정
                max_memory = {0: "24GB", 1: "24GB"}
            else:
                # GPU 1개인 경우 자동 배치
                logger.info("GPU 1개 자동 배치 설정")
                device_map = "auto"
                max_memory = {0: "24GB"}
            
            # 4-bit 양자화로 메모리 절약
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                load_in_4bit=True,  # 8bit 대신 4bit 사용
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("gemma-9b 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"4bit 모델 로딩 실패: {e}")
            # 8bit로 fallback
            logger.info("8bit 모드로 모델 로딩 시도...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_8bit=True,
                    max_memory=max_memory,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info("8bit 모드로 모델 로딩 완료")
            except Exception as e2:
                logger.error(f"8bit 모델 로딩도 실패: {e2}")
                # CPU 모드로 시도
                logger.info("CPU 모드로 모델 로딩 시도...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True
                )
                logger.info("CPU 모드로 모델 로딩 완료")
    
    def load_embeddings(self):
        """임베딩 모델 로드"""
        logger.info("임베딩 모델 로딩 중...")
        

        # GPU 개수에 따라 device 설정
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            device = "cuda:0"
            logger.info("임베딩 모델을 GPU 0에 배치")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"임베딩 모델을 {device}에 배치")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("임베딩 모델 로딩 완료")
        
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF 파일 읽기 오류 {pdf_path}: {e}")
            return ""
    
    def extract_text_from_xlsx(self, xlsx_path: str) -> str:
        """XLSX 파일에서 텍스트 추출"""
        try:
            df = pd.read_excel(xlsx_path)
            text = ""
            
            # 모든 컬럼의 텍스트를 결합
            for column in df.columns:
                text += f"{column}: " + " ".join(df[column].astype(str).dropna()) + "\n"
            
            return text
        except Exception as e:
            logger.error(f"XLSX 파일 읽기 오류 {xlsx_path}: {e}")
            return ""
    
    def load_documents(self, dataset_path: str = "dataset"):
        """데이터셋에서 문서 로드"""
        logger.info("문서 로딩 중...")
        dataset_dir = Path(dataset_path)
        
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_text_from_pdf(str(file_path))
                    if text.strip():
                        self.documents.append(Document(
                            page_content=text,
                            metadata={"source": str(file_path), "type": "pdf"}
                        ))
                        logger.info(f"PDF 로드 완료: {file_path.name}")
                
                elif file_path.suffix.lower() == '.xlsx':
                    text = self.extract_text_from_xlsx(str(file_path))
                    if text.strip():
                        self.documents.append(Document(
                            page_content=text,
                            metadata={"source": str(file_path), "type": "xlsx"}
                        ))
                        logger.info(f"XLSX 로드 완료: {file_path.name}")
        
        logger.info(f"총 {len(self.documents)}개 문서 로드 완료")
    
    def create_vectorstore(self):
        """벡터 스토어 생성"""
        if not self.documents:
            raise ValueError("문서가 로드되지 않았습니다.")
        
        logger.info("벡터 스토어 생성 중...")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        split_docs = text_splitter.split_documents(self.documents)
        logger.info(f"문서 분할 완료: {len(split_docs)}개 청크")
        
        # FAISS 벡터 스토어 생성
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
        # 벡터 스토어 저장
        self.vectorstore.save_local("vectorstore")
        logger.info("벡터 스토어 생성 및 저장 완료")
    
    def setup_retriever(self):
        """검색 설정 - 메모리 최적화"""
        logger.info("검색 시스템 설정 중...")
        
        try:
            # 벡터 검색만 사용 (메모리 절약)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # 검색 결과 수 줄임
            )
            logger.info("벡터 검색 설정 완료")
            
        except Exception as e:
            logger.error(f"검색 설정 실패: {e}")
            raise
    
    def generate_response(self, question: str, context: str) -> str:
        """응답 생성 - 메모리 최적화 버전"""
        try:
            # 컨텍스트 길이 제한 (메모리 절약)
            max_context_length = 4000
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            prompt = INITIAL_PROMPT + f"""다음은 임신과 출산에 관한 질문과 관련 정보입니다.

질문: {question}

관련 정보:
{context}

위의 정보를 바탕으로 질문에 대한 답변을 한국어로 작성해주세요. 
정보가 충분하지 않다면 "죄송합니다. 해당 내용에 대한 정보를 찾을 수 없습니다."라고 답변해주세요.

답변:"""

            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 입력 토크나이징 (attention_mask 명시적 설정)
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096,
                padding=True,
                return_attention_mask=True
            )
            
            # 모델이 여러 GPU에 분산된 경우 첫 번째 GPU 사용
            try:
                if hasattr(self.model, 'hf_device_map'):
                    device = list(self.model.hf_device_map.values())[0]
                    if isinstance(device, int):
                        device = f"cuda:{device}"
                else:
                    device = next(self.model.parameters()).device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"GPU 배치 실패, CPU 사용: {e}")
                inputs = {k: v.cpu() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 부분 제거
            response = response[len(prompt):].strip()
            
            # 메모리 정리
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
    def generate_response_stream(self, question: str, context: str):
        """효율적인 실시간 스트리밍 - TextIteratorStreamer 사용"""
        try:

            
            # 컨텍스트 길이 제한 (메모리 절약)
            max_context_length = 4000
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            prompt = INITIAL_PROMPT + f"""다음은 임신과 출산에 관한 질문과 관련 정보입니다.

질문: {question}

관련 정보:
{context}

위의 정보를 바탕으로 질문에 대한 답변을 한국어로 작성해주세요. 
정보가 충분하지 않다면 "죄송합니다. 해당 내용에 대한 정보를 찾을 수 없습니다."라고 답변해주세요.

답변:"""

            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 입력 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096,
                padding=True,
                return_attention_mask=True
            )
            
            # GPU 배치
            try:
                if hasattr(self.model, 'hf_device_map'):
                    device = list(self.model.hf_device_map.values())[0]
                    if isinstance(device, int):
                        device = f"cuda:{device}"
                else:
                    device = next(self.model.parameters()).device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                logger.warning(f"GPU 배치 실패, CPU 사용: {e}")
                inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # TextIteratorStreamer 설정
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0
            )
            
            # 생성 설정
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True
            }
            
            # 별도 스레드에서 생성 실행
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 스트리밍 결과 처리
            try:
                full_answer = ""
                for new_text in streamer:
                    if new_text:
                        full_answer += new_text
                        # 토큰별로 전송
                        yield f"data: {json.dumps({'token': new_text, 'finished': False}, ensure_ascii=False)}\n\n"

                # 최종 전체 답변 이벤트 전송
                yield f"data: {json.dumps({'type': 'final', 'answer': full_answer, 'finished': True}, ensure_ascii=False)}\n\n"
                
            except Exception as stream_error:
                logger.error(f"스트리밍 처리 오류: {stream_error}")
                yield f"data: {json.dumps({'token': '스트리밍 오류가 발생했습니다.', 'finished': True})}\n\n"
            
            finally:
                # 스레드 종료 대기
                thread.join(timeout=60.0)
                
                # 메모리 정리
                del inputs
                torch.cuda.empty_cache() 
            
        except Exception as e:
            logger.error(f"실시간 스트리밍 오류: {e}")
            yield f"data: {json.dumps({'token': '죄송합니다. 응답 생성 중 오류가 발생했습니다.', 'finished': True})}\n\n"
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        try:
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return {
                    "answer": "죄송합니다. 해당 내용에 대한 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 컨텍스트 생성
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            # 답변 생성
            answer = self.generate_response(question, context)
            
            # 신뢰도 계산
            confidence = min(0.9, len(docs) / 3.0)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0
            }
    
    def initialize(self):
        """전체 시스템 초기화"""
        logger.info("시스템 초기화 시작...")
        
        try:
            # 임베딩 모델 먼저 로드
            self.load_embeddings()
            logger.info("임베딩 모델 로드 완료")
            
            # 벡터 스토어 로딩 또는 생성
            self.load_or_create_vectorstore()
            logger.info("벡터 스토어 설정 완료")
            
            # 검색 설정
            self.setup_retriever()
            logger.info("검색 시스템 설정 완료")
            
            # 마지막에 대화 모델 로드
            self.load_model()
            logger.info("대화 모델 로드 완료")
            
            logger.info("시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            raise
    
    def load_or_create_vectorstore(self):
        """벡터 스토어 로딩 또는 생성"""
        vectorstore_path = "vectorstore"
        
        # 벡터 스토어가 이미 존재하는지 확인
        if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
            try:
                logger.info("기존 벡터 스토어를 로딩합니다...")
                self.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # 보안 경고 해결
                )
                logger.info("기존 벡터 스토어 로딩 완료")
                return
            except Exception as e:
                logger.warning(f"기존 벡터 스토어 로딩 실패: {e}")
                logger.info("새로운 벡터 스토어를 생성합니다...")
        
                # 벡터 스토어가 없거나 로딩 실패 시 새로 생성
                logger.info("새로운 벡터 스토어를 생성합니다...")
                self.load_documents()
                self.create_vectorstore()
                logger.info("벡터 스토어 생성 완료")

# 챗봇 인스턴스
chatbot = SimplePregnancyChatbot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 초기화
    print("Simple Chatbot is starting...")
    chatbot.initialize()
    yield
    print("Simple Chatbot is stopping...")

# FastAPI 앱 생성
app = FastAPI(title="임신과 출산 RAG 챗봇 (Simple)", version="0.1.0", lifespan=lifespan)

# CORS 설정: 외부 접근 허용 (필요시 특정 도메인으로 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        result = chatbot.answer_question(request.question)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"채팅 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다.")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 엔드포인트"""
    try:
        # 관련 문서 검색
        docs = chatbot.retriever.get_relevant_documents(request.question)
        
        if not docs:
            # 검색 결과가 없는 경우
            error_response = f"data: {json.dumps({'token': '죄송합니다. 해당 내용에 대한 정보를 찾을 수 없습니다.', 'finished': True})}\n\n"
            return StreamingResponse(
                iter([error_response]),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # 컨텍스트 생성
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 스트리밍 응답 생성
        def generate():
            # 소스 정보 먼저 전송
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            sources_info = f"data: {json.dumps({'sources': sources, 'type': 'sources'}, ensure_ascii=False)}\n\n"
            yield sources_info
            
            # 스트리밍 응답 생성
            for chunk in chatbot.generate_response_stream(request.question, context):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"스트리밍 채팅 오류: {e}")
        error_response = f"data: {json.dumps({'token': '서버 오류가 발생했습니다.', 'finished': True})}\n\n"
        return StreamingResponse(
            iter([error_response]),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "message": "임신과 출산 RAG 챗봇이 정상 작동 중입니다."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
