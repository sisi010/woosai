#!/usr/bin/env python3
# QLoRA 파인튜닝 스크립트 - 24GB VRAM 기준 최적화

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_qlora_config():
    """QLoRA 설정 - 24GB VRAM 기준"""
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4비트 양자화
        bnb_4bit_use_double_quant=True,       # 더블 양자화로 메모리 절약
        bnb_4bit_quant_type="nf4",            # 정규분포 4비트
        bnb_4bit_compute_dtype=torch.bfloat16  # 계산용 타입
    )
    return qlora_config

def setup_lora_config():
    """LoRA 어댑터 설정"""
    lora_config = LoraConfig(
        r=16,                    # 랭크 - 메모리와 성능의 균형점
        lora_alpha=32,           # 스케일링 파라미터 (일반적으로 r의 2배)
        target_modules=[         # Llama2/Mistral 기준 타겟 모듈
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,        # 드롭아웃 비율
        bias="none",             # 바이어스 학습 안함
        task_type="CAUSAL_LM"    # 인과적 언어모델
    )
    return lora_config

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    """모델과 토크나이저 로드"""
    qlora_config = setup_qlora_config()
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
    
    # 모델 로드 (4비트 양자화)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qlora_config,
        device_map="auto",           # GPU 자동 배치
        torch_dtype=torch.bfloat16,  # 메모리 효율성
        trust_remote_code=True
    )
    
    # LoRA를 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 어댑터 적용
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    logger.info(f"훈련 가능한 파라미터: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def format_instruction(example):
    """학습 데이터 포맷팅"""
    if example.get('input'):
        prompt = f"### 지시사항:\n{example['instruction']}\n\n### 입력:\n{example['input']}\n\n### 응답:\n{example['output']}"
    else:
        prompt = f"### 지시사항:\n{example['instruction']}\n\n### 응답:\n{example['output']}"
    
    return {"text": prompt}

def load_dataset_from_jsonl(data_path):
    """JSONL 파일에서 데이터셋 로드"""
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_path, "train.jsonl"),
        "validation": os.path.join(data_path, "valid.jsonl")
    })
    
    # 데이터 포맷팅
    dataset = dataset.map(format_instruction)
    
    logger.info(f"학습 데이터: {len(dataset['train'])}개")
    logger.info(f"검증 데이터: {len(dataset['validation'])}개")
    
    return dataset

def setup_training_args(output_dir="./checkpoints"):
    """학습 설정"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # 학습률과 스케줄러
        learning_rate=2e-4,              # QLoRA 권장 학습률
        lr_scheduler_type="cosine",      # 코사인 스케줄러
        warmup_steps=10,                 # 워밍업 단계
        
        # 배치 사이즈 (24GB 기준)
        per_device_train_batch_size=4,   # 실제 배치 사이즈
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   # 효과적 배치사이즈 = 4*4=16
        
        # 학습 단계
        num_train_epochs=3,              # 에포크 수
        max_steps=-1,                    # 자동 계산
        
        # 평가와 저장
        eval_strategy="steps",
        eval_steps=50,                   # 50스텝마다 평가
        save_strategy="steps",
        save_steps=100,                  # 100스텝마다 저장
        save_total_limit=2,              # 최대 2개 체크포인트 유지
        
        # 메모리 최적화
        fp16=False,                      # bfloat16 사용
        bf16=True,
        gradient_checkpointing=True,     # 메모리 절약
        dataloader_pin_memory=False,
        
        # 로깅
        logging_steps=10,
        report_to=None,                  # wandb 등 비활성화
        
        # 기타
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    return training_args

def main():
    """메인 학습 함수"""
    logger.info("QLoRA 파인튜닝 시작")
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    # 데이터 로드
    data_path = "../data"  # train.jsonl, valid.jsonl 위치
    dataset = load_dataset_from_jsonl(data_path)
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # 학습 설정
    training_args = setup_training_args()
    
    # SFT 트레이너 설정
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=setup_lora_config(),
        dataset_text_field="text",
        max_seq_length=512,              # 최대 시퀀스 길이
        tokenizer=tokenizer,
        args=training_args,
        packing=False,                   # 패킹 비활성화 (안정성)
    )
    
    # 학습 실행
    logger.info("학습 시작...")
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info(f"학습 완료! 모델 저장 위치: {training_args.output_dir}")
    
    # 메모리 정리
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()