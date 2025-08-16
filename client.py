import requests
import json
import time
import sys
from typing import Dict, Any

class PregnancyChatbotClient:
    def __init__(self, server_url: str = "http://localhost:9000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"서버 연결 오류: {e}"}
    
    def ask_question(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """질문 전송"""
        try:
            payload = {
                "question": question,
                "user_id": user_id
            }
            
            response = self.session.post(
                f"{self.server_url}/chat/stream",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"요청 오류: {e}"}
    
    def ask_question_stream(self, question: str, user_id: str = None):
        """스트리밍 질문 전송"""
        try:
            payload = {
                "question": question,
                "user_id": user_id
            }
            
            response = self.session.post(
                f"{self.server_url}/chat/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            return None

def main():
    """메인 함수 - 대화형 인터페이스"""
    client = PregnancyChatbotClient()
    
    print("=" * 50)
    print("임신과 출산 RAG 챗봇 클라이언트")
    print("=" * 50)
    
    # 서버 상태 확인
    health = client.health_check()
    if "error" in health:
        print(f"❌ 서버 연결 실패: {health['error']}")
        print("서버가 실행 중인지 확인해주세요.")
        return
    else:
        print(f"✅ 서버 연결 성공: {health['message']}")
    
    print("\n질문을 입력하세요. 'quit' 또는 'exit'를 입력하면 종료됩니다.")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("챗봇을 종료합니다.")
                break
            
            if not question:
                print("질문을 입력해주세요.")
                continue
            
            print("답변 생성 중...")
            start_time = time.time()
            
            # 스트리밍 모드로 답변 받기
            response = client.ask_question_stream(question)
            
            if response is None:
                print("❌ 스트리밍 연결 실패")
                continue
            
            print(f"\n🤖 답변 (실시간 스트리밍)")
            print("-" * 30)
            
            full_answer = ""
            sources = []
            
            try:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 'data: ' 제거
                            try:
                                data = json.loads(data_str)
                                
                                if data.get('type') == 'sources':
                                    sources = data.get('sources', [])
                                elif 'token' in data:
                                    token = data['token']
                                    if token:
                                        print(token, end='', flush=True)
                                        full_answer += token
                                    
                                    if data.get('finished', False):
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
                
                end_time = time.time()
                print(f"\n\n(응답시간: {end_time - start_time:.2f}초)")
                
                if sources:
                    print(f"\n📚 참고 자료:")
                    for i, source in enumerate(sources, 1):
                        print(f"  {i}. {source}")
                
                # 간단한 신뢰도 계산
                confidence = min(0.9, len(sources) / 3.0) if sources else 0.0
                print(f"\n신뢰도: {confidence:.2f}")
                
            except Exception as e:
                print(f"\n❌ 스트리밍 처리 오류: {e}")
                continue
            
        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()
