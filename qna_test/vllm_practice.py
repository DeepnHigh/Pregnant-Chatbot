# from vllm import LLM, SamplingParams

# # 모델 초기화
# llm = LLM(model="rtzr/ko-gemma-2-9b-it", tensor_parallel_size=2)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)

# # generate 메서드 사용 (입력을 리스트로 전달)
# outputs = llm.generate(["안녕하세요. 너의 소개를 해 줘"], sampling_params)

# # 결과 출력
# for output in outputs:
#     print(output.outputs[0].text)
#     print("---")

# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "rtzr/ko-gemma-2-9b-it",
#     device_map="auto",
#     low_cpu_mem_usage=True
# )

# model.save_pretrained("/home/dha8102/.cache/huggingface/hub/merged_ko-gemma2", safe_serialization=True)

# exit()

import multiprocessing as mp
mp.set_start_method("fork", force=True)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams

# 모델 및 FastAPI 초기화
app = FastAPI()
llm = LLM(model="/home/dha8102/.cache/huggingface/hub/merged_ko-gemma2", tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)

# 입력 포맷 (OpenAI style)
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "ko-gemma"
    messages: List[Message]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 500

# root 엔드포인트
@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI LLM chatbot is running."}


# OpenAI-style /v1/chat/completions 엔드포인트
@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages if msg.role != "system"])

    sampling = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )

    outputs = llm.generate([prompt], sampling)

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": outputs[0].outputs[0].text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }
