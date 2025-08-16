#!/usr/bin/env python3
"""
간단한 테스트 스크립트
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu():
    """GPU 상태 테스트"""
    print("=== GPU 상태 테스트 ===")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU 개수: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("CUDA를 사용할 수 없습니다.")
    print()

def test_embeddings():
    """임베딩 모델 테스트"""
    print("=== 임베딩 모델 테스트 ===")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 간단한 테스트
        test_text = "임신 중 운동은 언제부터 시작할 수 있나요?"
        embedding = embeddings.embed_query(test_text)
        print(f"임베딩 차원: {len(embedding)}")
        print("임베딩 모델 로드 성공!")
        
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
    print()

def test_model():
    """대화 모델 테스트"""
    print("=== 대화 모델 테스트 ===")
    try:
        # 더 작은 모델로 테스트
        model_name = "google/gemma-2b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        # 간단한 테스트
        prompt = "안녕하세요"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"모델 응답: {response}")
        print("대화 모델 로드 성공!")
        
    except Exception as e:
        print(f"대화 모델 로드 실패: {e}")
    print()

if __name__ == "__main__":
    test_gpu()
    test_embeddings()
    test_model()
    print("테스트 완료!")
