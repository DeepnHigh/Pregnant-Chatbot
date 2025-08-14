import pandas as pd
import faiss
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline



# 6. 질의 기반 응답 생성 함수
def ask_llm(user_input: str, k: int, label: int, model_name: str) -> tuple:
    input_embed = embed_model.encode([user_input], normalize_embeddings=True)
    distances, indices = index.search(input_embed, top_k)

    relevant_qas = []
    relevant_ind = []
    sim_score = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(questions) and score > 0.6:  # 코사인 유사도 0.6 이상만 사용
            relevant_qas.append(f"Q: {questions[idx]}\nA: {answers[idx]}")
            relevant_ind.append(idx)
            sim_score.append(score)
            
    if not relevant_qas:
        if label == 0:
            return "관련된 내용을 찾을 수 없습니다.", 1
        else:
            return "관련된 내용을 찾을 수 없습니다.", 0
    
    relevant_ind = np.array(relevant_ind)
    sort_ind = np.argsort(-np.array(sim_score))
    
    pred_ind = np.where(sort_ind < k)
    pred = relevant_ind[pred_ind]
    # print("relevant_ind:",relevant_ind)
    # print("sort_ind:",sort_ind)
    # print("pred_ind:",pred_ind)
    # print("pred:",pred)
    # print("label:",label)
    if label in pred+1:
        ans_cnt = 1
    else:
        ans_cnt = 0
        
    context = "\n\n".join(relevant_qas)
    prompt = pipe.tokenizer.apply_chat_template(
        [   {"role": "system", "content": f"당신은 산부인과 전문의입니다. 주어진 context를 참고하여 환자의 질문에 대해 간결하고 정확하게 답변하세요. 추가 질문이나 대화는 하지 마세요.\n\n context:\n{context}"},
            {"role": "user", "content": f"{user_input}\n"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if model_name == 'beomi/Llama-3-Open-Ko-8B':
        eos_ids = [pipe.tokenizer.eos_token_id]
        end_of_turn_id = pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if end_of_turn_id is not None:
            eos_ids.append(end_of_turn_id)
    else:
        eos_ids = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

    response = pipe(
        prompt,
        max_new_tokens=config["llm"]["max_new_tokens"],
        temperature=config["llm"]["temperature"],
        top_p=config["llm"]["top_p"],
        eos_token_id=eos_ids,
        do_sample=True,
        repetition_penalty=1.2
    )
    # print("prompt:")
    # print(prompt)
    # print("len(prompt):",len(prompt))
    # print("response:")
    # print(response)
    # print('response[0]["generated_text"][len(prompt):]:')
    # print(response[0]["generated_text"][len(prompt):])
    # quit()
    return response[0]["generated_text"][len(prompt):].strip(), ans_cnt

# 7. test
if __name__ == "__main__":
    
    # 1. 설정 불러오기
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_path = config['rag']['data_path']
    sheet = config['rag']['sheet_name']
    question_col = config['rag']['question_column']
    answer_col = config['rag']['answer_column']
    faiss_path = config['rag']['faiss_index_path']
    top_k = config['rag']['top_k']
    model_name = config["llm"]["model_id"]
    
    # 2. LLM 세팅
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "tensor_parallel_size": 2  # 2개의 GPU 사용
        },
        device_map="auto",
    )
    pipe.model.eval()

    # 3. 질의응답 데이터 불러오기
    df = pd.read_excel(data_path, sheet_name=sheet)
    questions = df[question_col].fillna("").tolist()
    answers = df[answer_col].fillna("").tolist()

    # 4. 임베딩 모델 준비
    embed_model_name = config["rag"]["embedding_model"]
    embed_model = SentenceTransformer(embed_model_name)
    
    # 5. FAISS 인덱스 준비 (Cosine 유사도 기반)
    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        print("FAISS index loaded.")
    else:
        print("Building FAISS index...")
        embeddings = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_path)
        print("FAISS index saved.")
        
    
    test_data_path = config['test']['data_path']
    test_sheet = config['test']['sheet_name']
    test_q = config['test']['question_column']
    original_q_ind = config['test']['original_q_ind']
    at_k = config['test']['at_k']
    save_path = config['test']['save_path']
    
    test_data = pd.read_excel(test_data_path, sheet_name=test_sheet)
    
    test_query = test_data[test_q].tolist()
    label = test_data[original_q_ind].tolist()
    test_len = len(test_query)
    
    ans = 0
    
    result = {"번호":[], "테스트 질문":[], "모델 답변":[], "정답여부":[], "정확도":[]}
    
    for i in tqdm(range(test_len)):
        pred, ans_cnt = ask_llm(test_query[i], at_k, label[i], model_name)
        
        # print(f"{i+1}번 질문:{test_query[i]}")
        # print(f"답변:{pred}")
        # print("-"*50)
        ans += ans_cnt
        
        result['번호'].append(i+1)
        result['테스트 질문'].append(test_query[i])
        result['모델 답변'].append(pred)
        result['정답여부'].append(ans_cnt)
        result['정확도'].append('')
        
    acc = ans / test_len
    print(f"acc:{acc:.2%}")
    result['정확도'][0] = f"acc:{acc:.2%}"
    result = pd.DataFrame(result)
    result.to_csv(save_path, encoding='utf-8-sig', index=False)    
