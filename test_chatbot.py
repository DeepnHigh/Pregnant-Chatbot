#!/usr/bin/env python3
"""
임신과 출산 RAG 챗봇 테스트 스크립트
"""

import requests
import json
import time
from typing import List, Dict

class ChatbotTester:
    def __init__(self, server_url: str = "http://localhost:9000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """헬스 체크 테스트"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                print("✅ 헬스 체크 성공")
                return True
            else:
                print(f"❌ 헬스 체크 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 헬스 체크 오류: {e}")
            return False
    
    def test_chat(self, question: str) -> Dict:
        """채팅 테스트"""
        try:
            payload = {"question": question}
            start_time = time.time()
            
            response = self.session.post(
                f"{self.server_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 질문: {question}")
                print(f"   답변: {result['answer'][:100]}...")
                print(f"   응답시간: {response_time:.2f}초")
                print(f"   신뢰도: {result['confidence']:.2f}")
                if result['sources']:
                    print(f"   참고자료: {len(result['sources'])}개")
                return result
            else:
                print(f"❌ 채팅 요청 실패: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ 채팅 테스트 오류: {e}")
            return {}

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("임신과 출산 RAG 챗봇 테스트")
    print("=" * 60)
    
    tester = ChatbotTester()
    
    # 헬스 체크
    if not tester.test_health():
        print("서버가 실행되지 않았습니다. 먼저 서버를 시작해주세요.")
        return
    
    print("\n" + "=" * 60)
    print("채팅 테스트 시작")
    print("=" * 60)
    
    # 테스트 질문들
    test_questions = [
        "임신 중 운동은 언제부터 시작할 수 있나요?",
        "임신 중 금기 식품은 무엇인가요?",
        "출산 후 산후조리는 어떻게 해야 하나요?",
        "난임 치료는 어떤 방법들이 있나요?",
        "임신 중 약물 복용 시 주의사항은 무엇인가요?",
        "태교는 언제부터 시작하면 좋나요?",
        "임신 중 체중 증가는 얼마나 되어야 하나요?",
        "분만 방법에는 어떤 것들이 있나요?",
        "임신 중 영양소 섭취는 어떻게 해야 하나요?",
        "산후 우울증은 어떻게 예방할 수 있나요?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i}/{len(test_questions)} ---")
        result = tester.test_chat(question)
        results.append({
            "question": question,
            "result": result,
            "success": bool(result)
        })
        time.sleep(1)  # 서버 부하 방지
    
    # 테스트 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"총 테스트: {total_tests}개")
    print(f"성공: {successful_tests}개")
    print(f"실패: {total_tests - successful_tests}개")
    print(f"성공률: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        avg_confidence = sum(r['result'].get('confidence', 0) for r in results if r['success']) / successful_tests
        print(f"평균 신뢰도: {avg_confidence:.2f}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    main()
