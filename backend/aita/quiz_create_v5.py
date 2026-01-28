'''
2025.12.29 /hlta/backend/aita/yaml 의 prompt 파일 사용 
2026.01.06 question_profile 추가, OpenAIEmbeddings(model="text-embedding-3-large" 추가
2026.01.07 yaml/ 이하 폴더명 수정(CLS, USR, SYS)
quiz_create_v2 단답형, 서술형 open
2026.01.21 quiz_create_v3 profile 사용여부(use_yn)에 따라 사용
2026.01.23 출제 요청 처리 구조 개선
        - custom_request에서 문항 수·유형·난이도 정보를 구조화하여 추출
        - 추출된 출제 계획을 프롬프트에 명시해 문항 수·유형 정확도 개선
        - 생성 결과 초과 시 백엔드 가드레일 적용
2026.01.28 quiz_create_v5 ChromaDB -> Milvus 전환 (RAG 참고 DB 변경)
'''
import os
import sys
import json
import asyncio
import yaml
import hashlib
import textwrap
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma  # Chroma 제거
# import chromadb  # Chroma 제거
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncpg
import socket

# 프로젝트 루트 경로 설정 (backend/)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# retriever 모듈 경로 추가
RETRIEVER_PATH = PROJECT_ROOT / 'retriever'
if str(RETRIEVER_PATH) not in sys.path:
    sys.path.insert(0, str(RETRIEVER_PATH))

from config import DATABASE_CONFIG

# Milvus Retriever 로드
try:
    from retriever import VectorDB as MilvusRetriever
    MILVUS_AVAILABLE = True
    print("Info: Milvus retriever (retriever.py) loaded successfully")
except ImportError as e:
    print(f"Warning: Milvus retriever not available: {e}")
    MILVUS_AVAILABLE = False

import random
import string

# DATABASE_CONFIG 호스트 보정 (host.docker.internal은 컨테이너 외부에서 접근 불가)
if DATABASE_CONFIG.get("host") == "host.docker.internal":
    try:
        socket.gethostbyname("host.docker.internal")
    except socket.gaierror:
        print("Info: DATABASE_HOST 'host.docker.internal'을 찾을 수 없어 'localhost'로 대체합니다.")
        DATABASE_CONFIG["host"] = "localhost"

YAML_DIR = BASE_DIR / "yaml"
DEFAULT_SYSTEM_PROMPT_PATH = YAML_DIR / "SYS" / "default.yaml"
DEFAULT_PERSONA = """당신은 전문적인 교육자이자 시험문제 출제자입니다.
제공된 교육 자료를 바탕으로 학습 목표에 맞는 고품질의 시험 문제를 생성합니다."""

# 환경 변수 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치해주세요.")

# 기본 모델 정의
class ExamQuestion(BaseModel):
    item_content: str = Field(description="시험 문제")
    item_type_cd: str = Field(description="문제 유형 코드 (MC: 객관식, BLK: 빈칸채우기, SA: 단답형, ESS: 서술형)")
    item_choices: Optional[Dict[str, str]] = Field(description="객관식 문제의 보기 (객관식인 경우에만 필요)", default=None)
    item_answer: str = Field(description="정답")
    item_explain: str = Field(description="문제 해설")
    item_diff_cd: str = Field(description="난이도 (H: 상, M: 중, E: 하)")
    item_id: Optional[str] = Field(description="문제 고유 ID", default=None)

class ExamOutput(BaseModel):
    questions: List[ExamQuestion]

    def to_dict(self):
        return {
            "questions": [question.model_dump() for question in self.questions]
        }

# 에이전트 1단계: 사용자 요청에서 출제 계획(문항 수·유형·난이도) 추출
class QuestionTypeCount(BaseModel):
    item_type_cd: str = Field(description="문제 유형 코드")
    item_diff_cd: Optional[str] = Field(
        default=None,
        description="해당 유형의 난이도 코드(H/M/E). 지정 안 하면 난이도 무관하게 계산."
    )
    count: int = Field(description="해당 유형·난이도 문항 수", ge=1)

class QuizPlan(BaseModel):
    requested_total: int = Field(description="총 출제할 문항 수 (정확히 이 개수만 생성)", ge=1, le=100)
    breakdown: List[QuestionTypeCount] = Field(
        description="유형·난이도별 문항 수. 각 항목의 count 합계는 requested_total과 가급적 일치해야 함."
    )
    difficulty: str = Field(description="전체 시험 대표 난이도 H(상)/M(중)/E(하). 혼합이면 M", default="M")
    other_notes: Optional[str] = Field(description="그 외 출제 시 반영할 참고사항", default=None)

# Milvus 연동 유틸리티 (Singleton)
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

async def get_relevant_docs_from_milvus(query: str, cls_id: str, top_k: int = 5):
    """Milvus에서 관련 문서 검색"""
    loop = asyncio.get_event_loop()
    vdb = await loop.run_in_executor(None, get_vdb)
    
    if not vdb:
        return []

    collection_name = os.getenv("COLLECTION_NAME", "hallym_dev9")
    
    try:
        # retriever.py의 retriever() 메서드 호출
        docs = await loop.run_in_executor(
            None,
            lambda: vdb.retriever(
                search_query=query,
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

def format_docs(docs):
    """문서 포맷팅"""
    if not docs:
        return None
    return '\n\n'.join([d.page_content for d in docs])

def load_yaml_file(path: Path) -> dict:
    try:
        if not path.exists():
            print(f"경고: YAML 파일을 찾을 수 없습니다: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        print(f"YAML 로드 중 오류 발생 ({path}): {str(e)}")
        return {}

def load_user_config(user_id: str) -> dict:
    yaml_path = YAML_DIR / "USR" / f"{user_id}.yaml"
    user_data = load_yaml_file(yaml_path)
    return {
        "persona": user_data.get("persona", DEFAULT_PERSONA)
    }

def load_cls_config(cls_id: str) -> dict:
    yaml_path = YAML_DIR / "CLS" / f"{cls_id}.yaml"
    cls_data = load_yaml_file(yaml_path)
    return {
        "subject_name": cls_data.get("subject_name", ""),
        "subject_characteristics": cls_data.get("subject_characteristics", "기본 교육과정 내용"),
        "question_profile": cls_data.get("question_profile", ""),
        "system_prompt": cls_data.get("system_prompt")
    }

async def get_profile_use_yn(profile_type: str, user_id: str | None = None, cls_id: str | None = None) -> str | None:
    profile_type = (profile_type or "").upper().strip()
    if profile_type not in {"USR", "CLS"}:
        return "Y"

    conn = None
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        if profile_type == "USR":
            if not user_id: return "Y"
            use_yn = await conn.fetchval("SELECT use_yn FROM aita_profile_mst WHERE profile_type = 'USR' AND user_id = $1", user_id)
        else:
            if not cls_id: return "Y"
            use_yn = await conn.fetchval("SELECT use_yn FROM aita_profile_mst WHERE profile_type = 'CLS' AND cls_id = $1", cls_id)
        if use_yn is None: return None
        use_yn = use_yn.strip().upper()
        return "Y" if use_yn not in {"Y", "N"} else use_yn
    except Exception as e:
        print(f"Warning: aita_profile_mst.use_yn 조회 실패: {e}")
        return "Y"
    finally:
        if conn: await conn.close()

PLAN_EXTRACT_MODEL = os.getenv("QUIZ_PLAN_EXTRACT_MODEL", "gpt-4o-mini")

def _format_quiz_plan_instruction(plan: QuizPlan) -> str:
    parts = [f"**총 문항 수: 정확히 {plan.requested_total}문항만 출제**합니다."]
    type_names = {"MC": "객관식", "BLK": "빈칸", "SA": "단답형", "ESS": "서술형", "OX": "O/X"}
    for b in plan.breakdown:
        name = type_names.get(b.item_type_cd.upper(), b.item_type_cd)
        diff = (b.item_diff_cd or plan.difficulty or "M").upper()
        parts.append(f"- {name} / 난이도 {diff}: {b.count}문항")
    parts.append(f"- 전체 대표 난이도: {plan.difficulty}")
    if plan.other_notes: parts.append(f"- 참고사항: {plan.other_notes.strip()}")
    return "\n".join(parts)

async def extract_quiz_plan(custom_request: str) -> Optional[QuizPlan]:
    if not (custom_request or "").strip(): return None
    plan_llm = ChatOpenAI(model=PLAN_EXTRACT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    plan_parser = JsonOutputParser(pydantic_object=QuizPlan)
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 퀴즈 출제 요청을 분석하는 역할입니다. JSON 형식으로만 응답하세요.\n\n{format_instructions}"),
        ("human", "{custom_request}")
    ])
    try:
        chain = plan_prompt | plan_llm | plan_parser
        out = await chain.ainvoke({"custom_request": custom_request, "format_instructions": plan_parser.get_format_instructions()})
        return QuizPlan(**out) if isinstance(out, dict) else out
    except Exception as e:
        print(f"Warning: extract_quiz_plan 실패: {e}")
        return None

def load_default_system_prompt() -> str:
    prompt_data = load_yaml_file(DEFAULT_SYSTEM_PROMPT_PATH)
    prompt = prompt_data.get("system_prompt_template")
    if prompt: return textwrap.dedent(prompt).strip()
    raise ValueError(f"시스템 프롬프트 로드 불가: {DEFAULT_SYSTEM_PROMPT_PATH}")

async def generate_exam(user_id: str, class_id: str, exam_config: dict):
    try:
        usr_use_yn = await get_profile_use_yn("USR", user_id=user_id)
        cls_use_yn = await get_profile_use_yn("CLS", cls_id=class_id)

        if usr_use_yn == "Y":
            user_config = load_user_config(user_id)
            persona = user_config["persona"]
        else:
            persona = DEFAULT_PERSONA

        if cls_use_yn == "Y":
            cls_config = load_cls_config(class_id)
            subject_name = cls_config.get("subject_name") or exam_config.get("subject_name", "")
            subject_characteristics = cls_config.get("subject_characteristics", "기본 교육과정 내용")
            question_profile = cls_config.get("question_profile", "")
            system_prompt_template = cls_config.get("system_prompt")
        else:
            subject_name = exam_config.get("subject_name", "")
            subject_characteristics = "기본 교육과정 내용"
            question_profile = ""
            system_prompt_template = None
        
        # 1. RAG 컨텍스트 준비 (Milvus 사용)
        custom_request = exam_config.get('custom_request', '없음')
        context = subject_characteristics
        
        if MILVUS_AVAILABLE:
            print(f"Info: Milvus에서 '{class_id}' 강의 자료 검색 중...")
            docs = await get_relevant_docs_from_milvus(query=custom_request, cls_id=class_id, top_k=5)
            formatted_docs = format_docs(docs)
            if formatted_docs:
                context = f"{subject_characteristics}\n\n관련 교육 자료:\n{formatted_docs}"
                print(f"Info: Milvus에서 {len(docs)}개의 문서를 찾았습니다.")
            else:
                print(f"Info: Milvus에서 관련 문서를 찾지 못했습니다.")

        plan = await extract_quiz_plan(custom_request)
        quiz_plan_instruction = _format_quiz_plan_instruction(plan) if plan else "사용자 요청에 따라 출제하세요."

        llm = ChatOpenAI(model="o1", api_key=OPENAI_API_KEY)
        output_parser = JsonOutputParser(pydantic_object=ExamOutput)
        format_instructions = output_parser.get_format_instructions()

        system_template = system_prompt_template or load_default_system_prompt()
        if question_profile:
            system_template += f"\n\n**출제 가이드라인:**\n{question_profile}"
        
        system_template += f"\n\n**【문항 수 준수】**\n{quiz_plan_instruction}"

        human_template = (
            "과목: {subject_name}\n출제 계획: {quiz_plan_instruction}\n"
            "사용자 요청: {custom_request}\n참고 자료:\n{context}\n\n"
            "위 내용을 바탕으로 정확히 지정된 문항 수만큼 한국어로 출제하세요."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        chain = prompt | llm | output_parser

        exam = await chain.ainvoke({
            "persona": persona,
            "context": context,
            "subject_name": subject_name,
            "custom_request": custom_request,
            "format_instructions": format_instructions,
            "quiz_plan_instruction": quiz_plan_instruction,
        })

        if isinstance(exam, dict): exam = ExamOutput(**exam)
        
        # 가드레일 (단순화된 버전)
        if plan and len(exam.questions) > plan.requested_total:
            exam.questions = exam.questions[:plan.requested_total]

        return {"exam_data": exam}
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def format_exam_for_printing(exam_data: ExamOutput):
    res = ["=" * 50]
    for i, q in enumerate(exam_data.questions, 1):
        res.append(f"{i}. [{q.item_type_cd}] {q.item_content}")
        if q.item_choices:
            for k, v in q.item_choices.items(): res.append(f"{k}. {v}")
        res.append(f"정답: {q.item_answer}\n해설: {q.item_explain}\n" + "-" * 40)
    return "\n".join(res)

def generate_item_id(course_id: str) -> str:
    return f"ITEM-{course_id}-" + ''.join(random.choices(string.ascii_uppercase, k=6))

async def save_quiz_to_db(exam_data: ExamOutput, session_id: str, chat_seq: int, course_id: str):
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        kst_time = datetime.now(timezone(timedelta(hours=9)))
        for q in exam_data.questions: q.item_id = generate_item_id(course_id)
        
        quiz_json = {
            "quiz_data": {"questions": [q.model_dump() for q in exam_data.questions]},
            "generated_at": kst_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(exam_data.questions)
        }
        await conn.execute("UPDATE aita_chatbot_session SET chat_file_json = $1, upd_dt = CURRENT_TIMESTAMP WHERE session_id = $2 AND chat_seq = $3",
                           json.dumps(quiz_json, ensure_ascii=False), session_id, chat_seq)
        return True
    finally:
        await conn.close()

async def quizmain(user_id: str, cls_id: str, session_id: str, chat_seq: int, exam_config: dict):
    res = await generate_exam(user_id, cls_id, exam_config)
    exam_data = res["exam_data"]
    formatted = await format_exam_for_printing(exam_data)
    await save_quiz_to_db(exam_data, session_id, chat_seq, exam_config.get('course_id', cls_id))
    return {"formatted_exam": formatted, "exam_data": exam_data}

if __name__ == "__main__":
    test_config = {"subject_name": "테스트", "custom_request": "O/X 문제 2개 내줘"}
    asyncio.get_event_loop().run_until_complete(quizmain("test_user", "2024-test", "sess_1", 0, test_config))
