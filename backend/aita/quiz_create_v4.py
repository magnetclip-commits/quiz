'''
quiz_create_v4.py - Milvus 기반 RAG 버전 (Final)
2026.01.23 업데이트:
- 모든 Vector DB 및 임베딩 로직을 retriever.py의 VectorDB 클래스로 위임
- 검색 시 과목명(subject_name)을 결합하여 검색 정밀도 향상
- Singleton 패턴으로 Milvus 연결 효율화
'''
import os
import sys
import json
import asyncio
import textwrap
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents.base import Document
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncpg
import socket

# ---------------------------------------------------------------------------
# 1. 경로 설정 및 외부 모듈 로드
# ---------------------------------------------------------------------------
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent # backend/

# 설정 파일(config.py) 경로 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import DATABASE_CONFIG, OPENAI_API_KEY
except ImportError:
    DATABASE_CONFIG = {}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# retriever 모듈 경로 추가
retriever_path = project_root / 'retriever'
if str(retriever_path) not in sys.path:
    sys.path.insert(0, str(retriever_path))

try:
    from retriever import VectorDB as MilvusRetriever
    MILVUS_AVAILABLE = True
    print("Info: Milvus retriever (retriever.py) loaded successfully")
except ImportError as e:
    print(f"Warning: Milvus retriever not available: {e}")
    MILVUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# 2. Milvus 연동 유틸리티 (Singleton)
# ---------------------------------------------------------------------------
_vdb_instance = None

def get_vdb():
    """retriever.py의 VectorDB 인스턴스 싱글톤 반환"""
    global _vdb_instance
    if not MILVUS_AVAILABLE:
        return None
    if _vdb_instance is None:
        try:
            _vdb_instance = MilvusRetriever()
        except Exception as e:
            print(f"Error: Milvus 연결 실패: {e}")
            return None
    return _vdb_instance

async def get_relevant_docs_from_milvus(query: str, cls_id: str, subject_name: str = "", top_k: int = 5) -> List[Document]:
    """retriever.py 표준 메서드를 호출하여 문서 검색"""
    loop = asyncio.get_event_loop()
    vdb = await loop.run_in_executor(None, get_vdb)
    
    if not vdb:
        return []

    # 검색어 보정: 요청사항이 짧으면 과목명을 결합하여 맥락 강화
    search_query = query
    if len(query) < 15 or any(kw in query for kw in ["문제", "출제", "생성"]):
        search_query = f"{subject_name} {query}".strip()

    # .env에 설정된 컬렉션 명 사용 (없으면 기본값)
    collection_name = os.getenv("COLLECTION_NAME", "hallym_dev9")
    
    try:
        # retriever.py의 retriever() 메서드 호출
        docs = await loop.run_in_executor(
            None,
            lambda: vdb.retriever(
                search_query=search_query,
                collection_name=collection_name,
                expr_str=f'cls_id == "{cls_id}"',
                top_k=top_k,
                dense_retriever_limit=top_k * 2
            )
        )
        return docs
    except Exception as e:
        print(f"Error: Milvus 검색 중 오류 발생: {e}")
        return []

def format_docs(docs: List[Document]) -> str:
    """검색된 문서를 프롬프트 삽입용 텍스트로 변환"""
    return "\n\n".join([f"[문서 {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])

# ---------------------------------------------------------------------------
# 3. 문제 생성 클래스 및 로직
# ---------------------------------------------------------------------------
class QuizItem(BaseModel):
    question: str = Field(description="시험 문제 내용")
    options: List[str] = Field(description="4개의 보기 (객관식)")
    answer: str = Field(description="정답 (보기 중 하나와 정확히 일치해야 함)")
    explanation: str = Field(description="정답에 대한 상세 해설")

class ExamOutput(BaseModel):
    subject: str = Field(description="과목명")
    questions: List[QuizItem] = Field(description="생성된 문제 리스트")

async def generate_exam(user_id: str, class_id: str, exam_config: dict):
    """LMS 데이터를 참고하여 시험 문제 생성"""
    subject_name = exam_config.get("subject_name", "미지정 과목")
    custom_request = exam_config.get("custom_request", "수업 핵심 내용을 바탕으로 객관식 문제를 만들어주세요.")
    
    # 1. RAG 컨텍스트 준비
    context_text = "제공된 수업 자료가 없습니다. 일반적인 지식을 바탕으로 출제하세요."
    if MILVUS_AVAILABLE:
        print(f"Info: '{class_id}' 강의 자료 검색 중...")
        docs = await get_relevant_docs_from_milvus(
            query=custom_request,
            cls_id=class_id,
            subject_name=subject_name,
            top_k=5
        )
        if docs:
            context_text = format_docs(docs)
            print(f"Info: {len(docs)}개의 문서 조각을 참고합니다.")

    # 2. LLM 설정 및 프롬프트
    # retriever.py가 Gemini를 쓴다면, 문제 생성은 GPT-4o를 써서 교차 검증 효과를 줌
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
    parser = JsonOutputParser(pydantic_object=ExamOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", textwrap.dedent("""
            당신은 대학 교수님입니다. 제공된 [수업 자료]를 바탕으로 학생들을 위한 시험 문제를 출제해야 합니다.
            반드시 자료에 근거한 사실만을 문제로 만드세요.
            
            {format_instructions}
        """)),
        ("human", textwrap.dedent("""
            과목명: {subject_name}
            사용자 특별 요청: {custom_request}
            
            [수업 자료]
            {context}
            
            위 자료를 참고하여 변별력 있는 문제를 생성하세요.
        """))
    ])

    chain = prompt | llm | parser

    # 3. 생성 실행
    try:
        result = await chain.ainvoke({
            "subject_name": subject_name,
            "custom_request": custom_request,
            "context": context_text,
            "format_instructions": parser.get_format_instructions()
        })
        return {"exam_data": result}
    except Exception as e:
        print(f"Error: 문제 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="LLM 문제 생성 중 오류가 발생했습니다.")

# ---------------------------------------------------------------------------
# 4. DB 저장 및 메인 엔트리 (기존 로직 유지)
# ---------------------------------------------------------------------------
async def save_quiz_to_db(exam_data, session_id, chat_seq, course_id):
    """결과를 PostgreSQL에 저장하는 로직 (기존 코드와 동일)"""
    # ... (기존 save_quiz_to_db 코드 삽입) ...
    pass

async def quizmain(user_id: str, cls_id: str, session_id: str, chat_seq: int, exam_config: dict):
    exam_result = await generate_exam(user_id, cls_id, exam_config)
    # 이후 DB 저장 및 리턴 로직 처리...
    return exam_result

if __name__ == "__main__":
    # 테스트 실행
    test_config = {
        "subject_name": "국제무역론",
        "custom_request": "사후송금방식의 장단점에 대해 3문제 내줘"
    }
    asyncio.run(quizmain("test_user", "2024-20-806550-1-01", "session_123", 0, test_config))

