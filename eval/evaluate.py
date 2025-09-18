#!/usr/bin/env python3
# 모델 평가 스크립트 - 정답률과 환각 체크

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from rouge_score import rouge_scorer
import logging
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, base_model_name: str, adapter_path: str = None):
        """평가기 초기화"""
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_model(self):
        """모델과 토크나이저 로드"""
        logger.info(f"모델 로드 중: {self.base_model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 베이스 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 어댑터가 있다면 로드
        if self.adapter_path and os.path.exists(self.adapter_path):
            logger.info(f"LoRA 어댑터 로드: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
        self.model.eval()
        logger.info("모델 로드 완료")
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """응답 생성"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 부분 제거
        response = response[len(prompt):].strip()
        return response
    
    def create_evaluation_dataset(self) -> List[Dict]:
        """평가용 데이터셋 생성 - 20문항"""
        eval_data = [
            # 요약 테스트 (5문항)
            {
                "type": "summary",
                "instruction": "다음 텍스트를 한 문장으로 요약해주세요.",
                "input": "기후변화는 지구의 평균 기온 상승으로 인한 전 지구적 현상입니다. 온실가스 배출 증가가 주요 원인이며, 해수면 상승, 극지방 빙하 감소, 이상기후 등의 결과를 가져옵니다.",
                "expected": "기후변화는 온실가스로 인한 지구 기온 상승 현상으로 해수면 상승과 이상기후를 유발합니다.",
                "keywords": ["기후변화", "온실가스", "기온상승"]
            },
            {
                "type": "summary", 
                "instruction": "다음 내용을 3줄로 요약해주세요.",
                "input": "인공지능 기술의 발달로 자율주행차가 현실화되고 있습니다. 센서와 카메라로 주변 환경을 인식하고, 딥러닝 알고리즘으로 상황을 판단합니다. 하지만 안전성과 윤리적 문제가 여전히 해결 과제입니다.",
                "expected": "인공지능으로 자율주행차가 현실화되고 있습니다.\n센서와 딥러닝으로 환경 인식 및 판단을 수행합니다.\n안전성과 윤리 문제가 해결 과제입니다.",
                "keywords": ["자율주행", "인공지능", "안전성"]
            },
            
            # 질의응답 테스트 (8문항)
            {
                "type": "qa",
                "instruction": "질문에 답변해주세요.",
                "input": "파이썬에서 리스트와 딕셔너리의 차이점은 무엇인가요?",
                "expected": "리스트는 순서가 있는 데이터를 저장하며 인덱스로 접근합니다. 딕셔너리는 키-값 쌍으로 데이터를 저장하며 키로 접근합니다.",
                "keywords": ["리스트", "딕셔너리", "인덱스", "키"]
            },
            {
                "type": "qa",
                "instruction": "질문에 답변해주세요.", 
                "input": "REST API의 주요 HTTP 메소드 4가지는 무엇인가요?",
                "expected": "GET(조회), POST(생성), PUT(수정), DELETE(삭제)가 주요 HTTP 메소드입니다.",
                "keywords": ["GET", "POST", "PUT", "DELETE"]
            },
            
            # 대화 테스트 (4문항)
            {
                "type": "chat",
                "instruction": "대화를 이어가세요.",
                "input": "안녕하세요! 프로그래밍을 배우고 싶은데 어떤 언어로 시작하는 게 좋을까요?",
                "expected": "안녕하세요! 첫 프로그래밍 언어로는 Python을 추천합니다.",
                "keywords": ["Python", "추천", "초보자"]
            },
            
            # 환각 체크 (3문항) - 사실이 아닌 내용 생성 방지
            {
                "type": "hallucination",
                "instruction": "질문에 답변해주세요.",
                "input": "2024년 노벨 물리학상 수상자는 누구인가요?",
                "expected": "죄송하지만 정확한 정보를 제공할 수 없습니다.",
                "anti_keywords": ["가상의 인물", "확실하지 않은 정보"]
            }
        ]
        
        return eval_data
    
    def evaluate_response(self, response: str, expected: str, keywords: List[str]) -> Dict:
        """응답 평가"""
        # ROUGE 점수 계산
        rouge_scores = self.rouge_scorer.score(expected, response)
        
        # 키워드 포함 여부 체크
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in response.lower())
        keyword_ratio = keyword_matches / len(keywords) if keywords else 0
        
        # 길이 적절성 (너무 짧거나 길지 않은지)
        length_score = 1.0 if 10 <= len(response) <= 500 else 0.5
        
        return {
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure, 
            "rougeL": rouge_scores['rougeL'].fmeasure,
            "keyword_ratio": keyword_ratio,
            "length_score": length_score,
            "response_length": len(response)
        }
    
    def run_evaluation(self) -> Dict:
        """전체 평가 실행"""
        if self.model is None:
            self.load_model()
            
        eval_dataset = self.create_evaluation_dataset()
        results = []
        
        logger.info(f"평가 시작 - 총 {len(eval_dataset)}개 문항")
        
        for i, item in enumerate(eval_dataset):
            logger.info(f"평가 진행: {i+1}/{len(eval_dataset)}")
            
            # 프롬프트 생성
            if item.get('input'):
                prompt = f"### 지시사항:\n{item['instruction']}\n\n### 입력:\n{item['input']}\n\n### 응답:\n"
            else:
                prompt = f"### 지시사항:\n{item['instruction']}\n\n### 응답:\n"
            
            # 응답 생성
            response = self.generate_response(prompt)
            
            # 평가
            eval_result = self.evaluate_response(
                response, 
                item['expected'], 
                item.get('keywords', [])
            )
            
            # 결과 저장
            result = {
                "question_id": i + 1,
                "type": item['type'],
                "prompt": prompt,
                "response": response,
                "expected": item['expected'],
                **eval_result
            }
            results.append(result)
        
        return self.calculate_summary_metrics(results)
    
    def calculate_summary_metrics(self, results: List[Dict]) -> Dict:
        """요약 지표 계산"""
        df = pd.DataFrame(results)
        
        summary = {
            "total_questions": len(results),
            "avg_rouge1": df['rouge1'].mean(),
            "avg_rouge2": df['rouge2'].mean(), 
            "avg_rougeL": df['rougeL'].mean(),
            "avg_keyword_ratio": df['keyword_ratio'].mean(),
            "avg_length_score": df['length_score'].mean(),
            "avg_response_length": df['response_length'].mean(),
            "by_type": df.groupby('type').agg({
                'rouge1': 'mean',
                'keyword_ratio': 'mean',
                'length_score': 'mean'
            }).to_dict()
        }
        
        # 상세 결과 저장
        df.to_csv("evaluation_results.csv", index=False)
        logger.info("상세 결과 저장: evaluation_results.csv")
        
        return summary

def main():
    """메인 실행 함수"""
    # 설정
    base_model_name = "microsoft/DialoGPT-medium"  # 실제 모델명으로 변경
    adapter_path = "../train/checkpoints"  # LoRA 어댑터 경로
    
    # 평가 실행
    evaluator = ModelEvaluator(base_model_name, adapter_path)
    results = evaluator.run_evaluation()
    
    # 결과 출력
    logger.info("=== 평가 결과 ===")
    logger.info(f"전체 문항수: {results['total_questions']}")
    logger.info(f"평균 ROUGE-1: {results['avg_rouge1']:.3f}")
    logger.info(f"평균 키워드 매칭: {results['avg_keyword_ratio']:.3f}")
    logger.info(f"평균 길이 점수: {results['avg_length_score']:.3f}")
    logger.info(f"평균 응답 길이: {results['avg_response_length']:.1f}자")
    
    # 타입별 결과
    for eval_type, scores in results['by_type'].items():
        logger.info(f"\n{eval_type} 타입:")
        logger.info(f"  ROUGE-1: {scores['rouge1']:.3f}")
        logger.info(f"  키워드 매칭: {scores['keyword_ratio']:.3f}")
    
    # JSON으로 결과 저장
    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("평가 완료! 결과 저장: evaluation_summary.json")

if __name__ == "__main__":
    main()