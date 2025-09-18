#!/usr/bin/env python3
# API 테스트 스크립트

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    print("=== 헬스 체크 테스트 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"상태: {response.status_code}")
    print(f"응답: {response.json()}")
    print()

def test_chat():
    """채팅 API 테스트"""
    print("=== 채팅 API 테스트 ===")
    
    payload = {
        "messages": [
            {"role": "user", "content": "안녕하세요! 파이썬에 대해 알려주세요."}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    print(f"상태: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"응답: {result['message']['content']}")
    else:
        print(f"오류: {response.text}")
    print()

def test_summarize():
    """요약 API 테스트"""
    print("=== 요약 API 테스트 ===")
    
    payload = {
        "text": "인공지능은 인간의 지능을 모방하여 학습하고 추론하는 컴퓨터 시스템입니다. 머신러닝과 딥러닝 기술을 통해 이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 활용되고 있습니다. 최근에는 생성형 AI가 주목받고 있으며, ChatGPT 같은 대화형 AI가 대표적입니다.",
        "max_length": 2,
        "format_type": "sentences"
    }
    
    response = requests.post(f"{BASE_URL}/v1/summarize", json=payload)
    print(f"상태: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"요약: {result['summary']}")
    else:
        print(f"오류: {response.text}")
    print()

def test_rag():
    """RAG 검색 API 테스트"""
    print("=== RAG 검색 API 테스트 ===")
    
    payload = {
        "query": "Python으로 웹 개발하는 방법",
        "top_k": 2,
        "include_source": True
    }
    
    response = requests.post(f"{BASE_URL}/v1/search", json=payload)
    print(f"상태: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"답변: {result['answer']}")
        print(f"출처: {len(result['sources'])}개")
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['title']} (점수: {source['score']:.3f})")
    else:
        print(f"오류: {response.text}")
    print()

def main():
    """모든 테스트 실행"""
    print(f"WoosAI API 테스트 시작 - {datetime.now()}")
    print(f"서버 주소: {BASE_URL}")
    print()
    
    try:
        test_health()
        test_chat()
        test_summarize()
        test_rag()
        
        print("✅ 모든 테스트 완료!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다.")
        print("먼저 'python run_server.py'로 서버를 시작하세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()