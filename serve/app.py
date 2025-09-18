#!/usr/bin/env python3
# FastAPI serving script with vLLM and RAG integration

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ML libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# OpenAI integration
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Data Models =====

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Timestamp")

class ChatRequest(BaseModel):
    """Chat request model"""
    messages: List[ChatMessage] = Field(..., description="Chat history")
    system_prompt: Optional[str] = Field(default=None, description="System prompt")
    max_tokens: int = Field(default=200, description="Maximum tokens")
    temperature: float = Field(default=0.7, description="Generation temperature")

class ChatResponse(BaseModel):
    """Chat response model"""
    message: ChatMessage
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str

class SummarizeRequest(BaseModel):
    """Summarization request model"""
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(default=3, description="Maximum number of sentences")
    format_type: str = Field(default="sentences", description="Output format: sentences, bullets, keywords")

class RAGSearchRequest(BaseModel):
    """RAG search request model"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=3, description="Top K documents")
    include_source: bool = Field(default=True, description="Include source information")

class RAGResponse(BaseModel):
    """RAG response model"""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: List[str]

# ===== Safety Filter =====

class SafetyFilter:
    """Safety filter for content checking"""
    
    def __init__(self):
        self.forbidden_keywords = [
            "violence", "terror", "suicide", "drugs", 
        ]
        
        import re
        self.pii_patterns = [
            re.compile(r'\d{3}-\d{4}-\d{4}'),
            re.compile(r'\d{6}-[1-4]\d{6}'),
            re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        ]
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """Content safety check"""
        issues = []
        
        for keyword in self.forbidden_keywords:
            if keyword in text:
                issues.append(f"Forbidden keyword found: {keyword}")
        
        for pattern in self.pii_patterns:
            if pattern.search(text):
                issues.append("Personal information pattern detected")
        
        if len(text) > 5000:
            issues.append("Text too long (exceeds 5000 characters)")
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues
        }

# ===== RAG System =====

class SimpleRAG:
    """Simple RAG system based on FAISS"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def load_documents(self, doc_path: str = "documents.json"):
        """Load documents and create embeddings"""
        logger.info(f"Loading documents: {doc_path}")
        
        if not os.path.exists(doc_path):
            sample_docs = [
                {
                    "id": 1,
                    "title": "Python Basics",
                    "content": "Python is a simple and readable programming language. It provides concepts like variables, functions, and classes.",
                    "category": "programming"
                },
                {
                    "id": 2, 
                    "title": "Machine Learning Introduction",
                    "content": "Machine learning is a technology that learns patterns from data. It is divided into supervised learning, unsupervised learning, and reinforcement learning.",
                    "category": "ai"
                },
                {
                    "id": 3,
                    "title": "FastAPI Development",
                    "content": "FastAPI is a Python web framework. It supports asynchronous processing and automatic documentation.",
                    "category": "web"
                }
            ]
            
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(sample_docs, f, ensure_ascii=False, indent=2)
            
            logger.info("Sample documents created")
        
        with open(doc_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        texts = []
        for doc in docs:
            self.documents.append(doc["content"])
            self.metadata.append({
                "id": doc["id"],
                "title": doc["title"], 
                "category": doc.get("category", "general")
            })
            texts.append(doc["content"])
        
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"RAG initialization complete - {len(docs)} documents, {dimension} dimensions")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search similar documents"""
        if self.index is None:
            raise ValueError("Documents not loaded.")
        
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx]
                })
        
        return results

# ===== Main Application =====

class WoosAIService:
    """WoosAI main service"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.safety_filter = SafetyFilter()
        self.rag_system = SimpleRAG()
        self.model_name = "woosai-v1"
        
        # OpenAI 설정 - 신버전 방식
        self.use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
        if self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                self.max_openai_tokens = int(os.getenv("MAX_OPENAI_TOKENS", "200"))
                logger.info(f"OpenAI integration enabled: {self.openai_model}")
            else:
                logger.warning("USE_OPENAI=true but OPENAI_API_KEY not found")
                self.use_openai = False
        
    async def initialize(self):
        """Service initialization"""
        logger.info("Initializing WoosAI service...")
        
        try:
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info(f"Model loaded successfully: {model_name}")
        except Exception as e:
            logger.warning(f"Model loading failed: {e}, running in basic response mode")
            self.tokenizer = None
            self.model = None
        
        self.rag_system.load_documents()
        logger.info("Service initialization complete")
    
    def _format_chat_prompt(self, messages: List[ChatMessage], system_prompt: Optional[str] = None) -> str:
        """Improved chat prompt - DialoGPT optimized"""
        conversation = ""
        
        # Use only recent 3 messages
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        for msg in recent_messages:
            if msg.role == "user":
                conversation += f"Human: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"AI: {msg.content}\n"
        
        conversation += "AI:"
        return conversation
    
    def _is_complex_query(self, message: str) -> bool:
        """복잡한 쿼리인지 판단하여 OpenAI 사용 여부 결정"""
        complex_indicators = [
            "explain", "analyze", "compare", "창작", "번역", "write", "create",
            "분석", "설명", "비교", "어떻게", "왜", "방법", "차이점"
        ]
        
        # 긴 메시지나 복잡한 키워드 포함시 OpenAI 사용
        if len(message) > 100:
            return True
        
        return any(indicator in message.lower() for indicator in complex_indicators)

    async def _openai_chat(self, request: ChatRequest) -> ChatResponse:
        """OpenAI API를 사용한 채팅 - 신버전"""
        try:
            # 메시지 포맷 변환
            messages = []
            
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful AI assistant named WoosAI. Provide concise, helpful responses in Korean when the user writes in Korean, and in English when they write in English."
                })
            
            # 최근 3개 메시지만 사용 (비용 절약)
            recent_messages = request.messages[-3:]
            for msg in recent_messages:
                messages.append({
                    "role": msg.role, 
                    "content": msg.content
                })
            
            # OpenAI API 호출 - 신버전
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=self.max_openai_tokens,
                temperature=request.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 응답 메시지 생성
            response_message = ChatMessage(
                role="assistant",
                content=response_text,
                timestamp=datetime.now().isoformat()
            )
            
            return ChatResponse(
                message=response_message,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                },
                model=f"{self.model_name}-openai"
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # OpenAI 실패시 로컬 모델로 fallback
            return await self._local_chat(request)

    async def _local_chat(self, request: ChatRequest) -> ChatResponse:
        """로컬 모델을 사용한 채팅"""
        last_message = request.messages[-1].content if request.messages else ""
        prompt = self._format_chat_prompt(request.messages, request.system_prompt)
        
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 30,
                        do_sample=True,
                        temperature=min(request.temperature, 0.8),
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                
                response_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                response_text = response_text.replace("Human:", "").replace("AI:", "").strip()
                response_text = response_text.split('\n')[0].strip()

                if len(response_text) < 3:
                    response_text = "I understand your question, but I need more context to provide a helpful response."
                elif len(response_text) > 200:
                    response_text = response_text[:200].rstrip() + "..."
                    
            except Exception as e:
                logger.error(f"Local model inference error: {e}")
                response_text = f"I'd like to help with '{last_message}' but encountered a processing issue."
        else:
            response_text = f"Hello! I'm preparing a response to '{last_message}'. Model is loading..."
        
        response_message = ChatMessage(
            role="assistant",
            content=response_text,
            timestamp=datetime.now().isoformat()
        )
        
        return ChatResponse(
            message=response_message,
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(response_text)},
            model=f"{self.model_name}-local"
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """개선된 채팅 - OpenAI와 로컬 모델 조건부 사용"""
        # 안전성 검사
        last_message = request.messages[-1].content if request.messages else ""
        safety_check = self.safety_filter.check_content(last_message)
        
        if not safety_check["is_safe"]:
            raise HTTPException(status_code=400, detail=f"Unsafe content: {safety_check['issues']}")
        
        # OpenAI 사용 여부 결정
        if self.use_openai and self._is_complex_query(last_message):
            logger.info(f"Using OpenAI for complex query: {last_message[:50]}...")
            return await self._openai_chat(request)
        else:
            logger.info(f"Using local model for simple query: {last_message[:50]}...")
            return await self._local_chat(request)

    async def _openai_summarize(self, request: SummarizeRequest) -> str:
        """OpenAI를 사용한 요약 - 신버전"""
        try:
            if request.format_type == "bullets":
                instruction = f"Summarize the following text in {request.max_length} bullet points:"
            elif request.format_type == "keywords":
                instruction = f"Extract {request.max_length} key keywords from the following text:"
            else:
                instruction = f"Summarize the following text in {request.max_length} sentences:"
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful summarization assistant."},
                    {"role": "user", "content": f"{instruction}\n\n{request.text}"}
                ],
                max_tokens=self.max_openai_tokens,
                temperature=0.3  # 요약은 더 일관되게
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI summarization error: {e}")
            return "Error occurred while processing the summary with OpenAI."

    async def summarize(self, request: SummarizeRequest) -> Dict[str, str]:
        """개선된 요약 - OpenAI 우선 사용"""
        safety_check = self.safety_filter.check_content(request.text)
        if not safety_check["is_safe"]:
            raise HTTPException(status_code=400, detail=f"Unsafe content: {safety_check['issues']}")
        
        # 긴 텍스트는 항상 OpenAI 사용
        if self.use_openai and len(request.text) > 200:
            summary = await self._openai_summarize(request)
        else:
            # 로컬 모델 사용 (기존 로직)
            if request.format_type == "bullets":
                format_instruction = f"Summarize the following text in {request.max_length} bullet points."
            elif request.format_type == "keywords":
                format_instruction = f"Extract {request.max_length} key keywords from the following text."
            else:
                format_instruction = f"Summarize the following text in {request.max_length} sentences."
            
            # 텍스트 길이 제한
            truncated_text = request.text[:800]
            prompt = f"Text: {truncated_text}\n{format_instruction}\nSummary: "
            
            if self.model is not None and self.tokenizer is not None:
                try:
                    inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_length=inputs.shape[1] + 50,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    summary = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                    if not summary:
                        summary = "Cannot generate summary due to text length limitations."
                except Exception as e:
                    logger.error(f"Summarization error: {e}")
                    summary = "Error occurred while processing the summary."
            else:
                summary = "Summary model is loading..."
        
        return {"summary": summary}
    
    async def rag_search(self, request: RAGSearchRequest) -> RAGResponse:
        """RAG question answering"""
        safety_check = self.safety_filter.check_content(request.query)
        if not safety_check["is_safe"]:
            raise HTTPException(status_code=400, detail=f"Unsafe content: {safety_check['issues']}")
        
        search_results = self.rag_system.search(request.query, request.top_k)
        contexts = [result["content"] for result in search_results]
        context_text = "\n\n".join(contexts)
        
        rag_prompt = f"""Instruction: Answer the question based on the context below.

Context:
{context_text}

Question: {request.query}

Answer: """
        
        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer.encode(rag_prompt, return_tensors="pt", max_length=400, truncation=True)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 80,
                        do_sample=True,
                        temperature=0.6,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                answer = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                if not answer:
                    answer = f"Cannot find information about '{request.query}'."
            except Exception as e:
                logger.error(f"RAG answer generation error: {e}")
                answer = "Error occurred while processing the search results."
        else:
            answer = "Search model is loading..."
        
        sources = []
        if request.include_source:
            for result in search_results:
                sources.append({
                    "title": result["metadata"]["title"],
                    "category": result["metadata"]["category"],
                    "score": result["score"]
                })
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=contexts
        )

# ===== FastAPI App =====

app = FastAPI(
    title="WoosAI API",
    description="Small Language AI Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

woosai_service = WoosAIService()

@app.on_event("startup")
async def startup_event():
    await woosai_service.initialize()

@app.get("/")
async def root():
    return {
        "service": "WoosAI",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": woosai_service.tokenizer is not None,
        "rag_ready": woosai_service.rag_system.index is not None,
        "openai_enabled": woosai_service.use_openai,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        return await woosai_service.chat(request)
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/summarize")
async def summarize_text(request: SummarizeRequest):
    try:
        return await woosai_service.summarize(request)
    except Exception as e:
        logger.error(f"Summarization processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/search", response_model=RAGResponse)
async def rag_search(request: RAGSearchRequest):
    try:
        return await woosai_service.rag_search(request)
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": woosai_service.model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "woosai"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )