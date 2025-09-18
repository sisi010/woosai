#!/usr/bin/env python3
# 서버 실행 스크립트

import subprocess
import sys
import os

def check_requirements():
    """필수 패키지 설치 확인"""
    try:
        import uvicorn
        import fastapi
        print("✅ FastAPI 설치 확인됨")
    except ImportError:
        print("❌ FastAPI가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install -r infra/requirements.txt")
        sys.exit(1)

def main():
    """메인 실행 함수"""
    print("WoosAI 서버 시작 중...")
    
    # 요구사항 확인
    check_requirements()
    
    # 서버 디렉토리로 이동
    serve_dir = os.path.join(os.path.dirname(__file__), "serve")
    os.chdir(serve_dir)
    
    # 서버 실행
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", "0.0.0.0",
        "--port", "8000", 
        "--reload",
        "--log-level", "info"
    ]
    
    print("서버 주소: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("Ctrl+C로 종료")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n서버 종료됨")

if __name__ == "__main__":
    main()