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
from langchain_chroma import Chroma
import chromadb
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncpg
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tutor.config import DATABASE_CONFIG
from config import DATABASE_CONFIG
import random
import string
import socket

# DATABASE_CONFIG 호스트 보정 (host.docker.internal은 컨테이너 외부에서 접근 불가)
if DATABASE_CONFIG.get("host") == "host.docker.internal":
    try:
        socket.gethostbyname("host.docker.internal")
    except socket.gaierror:
        print("Info: DATABASE_HOST 'host.docker.internal'을 찾을 수 없어 'localhost'로 대체합니다.")
        DATABASE_CONFIG["host"] = "localhost"

BASE_DIR = Path(__file__).resolve().parent
YAML_DIR = BASE_DIR / "yaml"
DEFAULT_SYSTEM_PROMPT_PATH = YAML_DIR / "SYS" / "default.yaml"
DEFAULT_PERSONA = """당신은 전문적인 교육자이자 시험문제 출제자입니다.
제공된 교육 자료를 바탕으로 학습 목표에 맞는 고품질의 시험 문제를 생성합니다."""

# 환경 변수 설정
# .env 파일에 있는 환경 변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
# CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))
# CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
CHROMA_HOST = os.getenv("CHROMA_HOST", "hlta-chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

try:
    # 1. 시도: Docker 내부 네트워크용 주소
    print(f"Info: ChromaDB 연결 시도 중 ({CHROMA_HOST}:{CHROMA_PORT})...")
    CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    # 실제 연결 확인을 위해 간단한 호출 시도
    CHROMA_CLIENT.heartbeat()
except Exception:
    try:
        # 2. 시도: 로컬 개발 환경용 주소 (호스트 포트 8002)
        LOCAL_HOST = "127.0.0.1"
        LOCAL_PORT = 8002
        print(f"Info: Docker 네트워크 연결 실패. 로컬 환경 연결 시도 중 ({LOCAL_HOST}:{LOCAL_PORT})...")
        CHROMA_CLIENT = chromadb.HttpClient(host=LOCAL_HOST, port=LOCAL_PORT)
        CHROMA_CLIENT.heartbeat()
    except Exception as e:
        print(f"Warning: 모든 ChromaDB 연결 시도가 실패했습니다. 검색 기능이 제한됩니다.")
        CHROMA_CLIENT = None

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
    """유형별 문항 수. item_type_cd: MC(객관식), BLK(빈칸), SA(단답), ESS(서술), OX(O/X)"""
    item_type_cd: str = Field(description="문제 유형 코드")
    count: int = Field(description="해당 유형 문항 수", ge=1)


class QuizPlan(BaseModel):
    """퀴즈 출제 요청에서 추출한 구조화된 계획. 문항 수·유형·난이도를 명확히 함."""
    requested_total: int = Field(description="총 출제할 문항 수 (정확히 이 개수만 생성)", ge=1, le=100)
    breakdown: List[QuestionTypeCount] = Field(
        description="유형별 문항 수. breakdown 내 count 합계는 requested_total과 일치해야 함."
    )
    difficulty: str = Field(description="난이도 H(상)/M(중)/E(하). 혼합이면 M", default="M")
    other_notes: Optional[str] = Field(description="그 외 출제 시 반영할 참고사항", default=None)

# Vectorstore 관련 함수들
def get_chroma_vectorstore(cls_id: str):
    """Chroma vectorstore 초기화 및 반환"""
    try:
        if CHROMA_CLIENT is None:
            return None
        #chroma_db_path = os.path.join("./db/chromadb", cls_id)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            #persist_directory=chroma_db_path,
            client=CHROMA_CLIENT, # 서버방식으로 변경
            embedding_function=embeddings_model,
            collection_name=cls_id
        )
        return vectorstore
    except Exception as e:
        print(f"ChromaDB 초기화 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def get_vectorstore(cls_id: str):
    """비동기적으로 vectorstore 가져오기"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_chroma_vectorstore, cls_id)

async def get_relevant_docs(retriever, query: str):
    """관련 문서 검색"""
    try:
        if retriever is None:
            return None
        
        return await retriever.ainvoke(query)
    except Exception as e:
        print(f"문서 검색 중 오류 발생: {str(e)}")
        return None

def format_docs(docs):
    """문서 포맷팅"""
    if not docs:
        return None
    return '\n\n'.join([d.page_content for d in docs])

def load_yaml_file(path: Path) -> dict:
    """일반 yaml 로더: 파일 없거나 오류 시 빈 dict 반환"""
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
    """사용자별 설정 정보를 YAML 파일에서 로드 (yaml/USR/{user_id}.yaml)"""
    yaml_path = YAML_DIR / "USR" / f"{user_id}.yaml"
    user_data = load_yaml_file(yaml_path)

    return {
        "persona": user_data.get("persona", DEFAULT_PERSONA)
    }


def load_cls_config(cls_id: str) -> dict:
    """클래스별 설정 정보를 YAML 파일에서 로드 (yaml/CLS/{cls_id}.yaml)"""
    yaml_path = YAML_DIR / "CLS" / f"{cls_id}.yaml"
    cls_data = load_yaml_file(yaml_path)

    return {
        "subject_name": cls_data.get("subject_name", ""),
        "subject_characteristics": cls_data.get("subject_characteristics", "기본 교육과정 내용"),
        "question_profile": cls_data.get("question_profile", ""),
        "system_prompt": cls_data.get("system_prompt")
    }

async def get_profile_use_yn(profile_type: str, user_id: str | None = None, cls_id: str | None = None) -> str | None:
    """
    aita_profile_mst에서 프로필 사용여부(use_yn)를 조회한다.

    요구사항:
    - profile_type='USR' 이면 user_id 조건으로 조회
    - profile_type='CLS' 이면 cls_id 조건으로 조회

    반환값:
    - 'Y' 또는 'N': 레코드가 있고 use_yn 값
    - None: 레코드가 없음
    - DB 오류 시 기존 동작을 깨지 않도록 기본값 'Y'를 반환한다.
    """
    profile_type = (profile_type or "").upper().strip()
    if profile_type not in {"USR", "CLS"}:
        return "Y"

    conn = None
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        if profile_type == "USR":
            if not user_id:
                return "Y"
            use_yn = await conn.fetchval(
                """
                SELECT use_yn
                FROM aita_profile_mst
                WHERE profile_type = 'USR'
                  AND user_id = $1
                """,
                user_id,
            )
        else:  # CLS
            if not cls_id:
                return "Y"
            use_yn = await conn.fetchval(
                """
                SELECT use_yn
                FROM aita_profile_mst
                WHERE profile_type = 'CLS'
                  AND cls_id = $1
                """,
                cls_id,
            )

        # 레코드가 없으면 None 반환
        if use_yn is None:
            return None

        use_yn = use_yn.strip().upper()
        return "Y" if use_yn not in {"Y", "N"} else use_yn
    except Exception as e:
        print(f"Warning: aita_profile_mst.use_yn 조회 실패(profile_type={profile_type}, user_id={user_id}, cls_id={cls_id}): {e}")
        return "Y"
    finally:
        if conn:
            await conn.close()


# 계획 추출용 경량 LLM (빠른 JSON 응답)
PLAN_EXTRACT_MODEL = os.getenv("QUIZ_PLAN_EXTRACT_MODEL", "gpt-4o-mini")


def _format_quiz_plan_instruction(plan: QuizPlan) -> str:
    """QuizPlan → LLM에 넘길 '문항 수·유형 필수 준수' 지시 문자열 생성."""
    parts = [
        f"**총 문항 수: 정확히 {plan.requested_total}문항만 출제**합니다. "
        f"{plan.requested_total - 1}문항이나 {plan.requested_total + 1}문항을 내면 안 됩니다."
    ]
    type_names = {
        "MC": "객관식(MC)",
        "BLK": "빈칸채우기(BLK)",
        "SA": "단답형(SA)",
        "ESS": "서술형(ESS)",
        "OX": "O/X(OX)",
    }
    for b in plan.breakdown:
        name = type_names.get(b.item_type_cd.upper(), b.item_type_cd)
        parts.append(f"- {name}: {b.count}문항")
    parts.append(f"- 난이도: {plan.difficulty} (H/M/E)")
    if plan.other_notes and plan.other_notes.strip():
        parts.append(f"- 참고사항: {plan.other_notes.strip()}")
    return "\n".join(parts)


async def extract_quiz_plan(custom_request: str) -> Optional[QuizPlan]:
    """
    사용자 요청(custom_request)에서 문항 수·유형·난이도를 구조화해 추출.
    에이전트 1단계: 이 계획을 바탕으로 2단계에서 '정확히 N문항' 출제 지시.
    """
    if not (custom_request or "").strip():
        return None

    plan_llm = ChatOpenAI(
        model=PLAN_EXTRACT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    plan_parser = JsonOutputParser(pydantic_object=QuizPlan)

    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 퀴즈 출제 요청을 분석하는 역할입니다.
사용자 요청에서 **총 문항 수(requested_total)**, **유형별 문항 수(breakdown)**, **난이도(difficulty)**를 추출해 아래 형식의 JSON만 출력하세요.

규칙:
- requested_total: 사용자가 요청한 총 문제 개수. 명시 안 되면 5로 추정.
- breakdown: 유형별 개수. item_type_cd는 MC/BLK/SA/ESS/OX 중 하나. breakdown의 count 합계 = requested_total.
- "객관식 20문제" → requested_total=20, breakdown=[{{"item_type_cd":"MC","count":20}}]
- "객관식 15, 빈칸 5" → requested_total=20, breakdown=[{{"item_type_cd":"MC","count":15}},{{"item_type_cd":"BLK","count":5}}]
- 유형 미명시 시 객관식(MC)로 간주.
- difficulty: H(상)/M(중)/E(하). 미명시 시 M.

{format_instructions}"""),
        ("human", "다음 출제 요청을 분석하세요.\n\n{custom_request}"),
    ])

    try:
        chain = plan_prompt | plan_llm | plan_parser
        out = await chain.ainvoke({
            "custom_request": custom_request,
            "format_instructions": plan_parser.get_format_instructions(),
        })
        plan = QuizPlan(**out) if isinstance(out, dict) else out

        # breakdown 합계 보정: requested_total과 다르면 단일 유형(MC)으로 통일
        total_from_breakdown = sum(b.count for b in plan.breakdown)
        if total_from_breakdown != plan.requested_total or not plan.breakdown:
            plan = QuizPlan(
                requested_total=plan.requested_total,
                breakdown=[QuestionTypeCount(item_type_cd="MC", count=plan.requested_total)],
                difficulty=plan.difficulty,
                other_notes=plan.other_notes,
            )
        return plan
    except Exception as e:
        print(f"Warning: extract_quiz_plan 실패 (custom_request 일부 사용): {e}")
        return None


def load_default_system_prompt() -> str:
    """기본 시스템 프롬프트를 YAML에서 로드"""
    prompt_data = load_yaml_file(DEFAULT_SYSTEM_PROMPT_PATH)
    prompt = prompt_data.get("system_prompt_template")
    if prompt:
        return textwrap.dedent(prompt).strip()
    
    # YAML 파일이 없거나 system_prompt_template 키가 없는 경우 에러 발생
    raise ValueError(
        f"시스템 프롬프트를 로드할 수 없습니다. "
        f"YAML 파일을 확인해주세요: {DEFAULT_SYSTEM_PROMPT_PATH}"
    )


async def generate_exam(user_id: str, class_id: str, exam_config: dict):
    """시험지 생성 함수"""
    try:
        # 프로필(YAML) 사용여부 조회
        usr_use_yn = await get_profile_use_yn("USR", user_id=user_id)
        cls_use_yn = await get_profile_use_yn("CLS", cls_id=class_id)

        # 사용자별 설정 로드 (use_yn='N'이면 YAML 미참조)
        usr_yaml_path = YAML_DIR / "USR" / f"{user_id}.yaml"
        if usr_use_yn == "Y":
            usr_yaml_exists = usr_yaml_path.exists()
            user_config = load_user_config(user_id)
            persona = user_config["persona"]
            if usr_yaml_exists:
                print(f"Info: 사용자 프로필 YAML 파일을 참고합니다: {usr_yaml_path}")
            else:
                print(f"Info: 사용자 프로필 YAML 파일이 없어 기본값을 사용합니다 (use_yn=Y이지만 파일 없음): {usr_yaml_path}")
        elif usr_use_yn == "N":
            persona = DEFAULT_PERSONA
            print(f"Info: 사용자 프로필 YAML 파일을 참고하지 않습니다 (use_yn=N): {usr_yaml_path}")
        else:  # None (레코드 없음)
            persona = DEFAULT_PERSONA
            print(f"Info: 사용자 프로필 YAML 파일을 참고하지 않습니다 (DB에 레코드 없음): {usr_yaml_path}")

        # 클래스별 설정 로드 (use_yn='N'이면 YAML 미참조)
        cls_yaml_path = YAML_DIR / "CLS" / f"{class_id}.yaml"
        if cls_use_yn == "Y":
            cls_yaml_exists = cls_yaml_path.exists()
            cls_config = load_cls_config(class_id)
            subject_name = cls_config.get("subject_name") or exam_config.get("subject_name", "")
            subject_characteristics = cls_config.get("subject_characteristics", "기본 교육과정 내용")
            question_profile = cls_config.get("question_profile", "")
            system_prompt_template = cls_config.get("system_prompt")
            if cls_yaml_exists:
                print(f"Info: 클래스 프로필 YAML 파일을 참고합니다: {cls_yaml_path}")
            else:
                print(f"Info: 클래스 프로필 YAML 파일이 없어 기본값을 사용합니다 (use_yn=Y이지만 파일 없음): {cls_yaml_path}")
        elif cls_use_yn == "N":
            subject_name = exam_config.get("subject_name", "")
            subject_characteristics = "기본 교육과정 내용"
            question_profile = ""
            system_prompt_template = None
            print(f"Info: 클래스 프로필 YAML 파일을 참고하지 않습니다 (use_yn=N): {cls_yaml_path}")
        else:  # None (레코드 없음)
            subject_name = exam_config.get("subject_name", "")
            subject_characteristics = "기본 교육과정 내용"
            question_profile = ""
            system_prompt_template = None
            print(f"Info: 클래스 프로필 YAML 파일을 참고하지 않습니다 (DB에 레코드 없음): {cls_yaml_path}")
        
        vectorstore = await get_vectorstore(class_id)
        # vectorstore = None  # 임시로 비활성화 mjo 
        context = subject_characteristics  # 과목 특성을 기본 context로 설정
        
        if vectorstore is not None:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "lambda_mult": 0.7}
            )
            
            # 관련 문서 검색
            custom_request = exam_config.get('custom_request', '없음')
            docs = await get_relevant_docs(retriever, custom_request)
            
            # 문서가 있는 경우에만 context 업데이트
            formatted_docs = format_docs(docs)
            if formatted_docs is not None:
                context = f"{subject_characteristics}\n\n관련 교육 자료:\n{formatted_docs}"
                print(f"Info: 관련 문서를 찾았습니다.")
            else:
                print(f"Info: {class_id} 강의에서 '{custom_request}'와 관련된 문서를 찾을 수 없어 문서를 참고하지 않고 시험지를 생성합니다.")

        custom_request = exam_config.get('custom_request', '없음')
        # 에이전트 1단계: 요청에서 문항 수·유형·난이도 구조화 추출 → 2단계에서 '정확히 N문항' 지시
        plan = await extract_quiz_plan(custom_request)
        if plan:
            quiz_plan_instruction = _format_quiz_plan_instruction(plan)
            print(f"Info: 출제 계획 추출됨 — 총 {plan.requested_total}문항, 유형별: {[(b.item_type_cd, b.count) for b in plan.breakdown]}")
        else:
            quiz_plan_instruction = (
                "문항 수·유형은 사용자 요청(아래 custom_request)에 명시된 대로 **정확히** 지키세요. "
                "1문항이라도 많거나 적으면 안 됩니다."
            )

        llm = ChatOpenAI(
            model="o1",
            api_key=OPENAI_API_KEY
        )

        # JsonOutputParser를 사용한 형식 지시사항 생성
        output_parser = JsonOutputParser(pydantic_object=ExamOutput)
        format_instructions = output_parser.get_format_instructions()

        # 시험 문제 출제 시스템 프롬프트 (yaml에서 가져오거나 기본값 사용)
        if system_prompt_template:
            system_template = textwrap.dedent(system_prompt_template).strip()
            print(f"Info: 시스템 프롬프트를 클래스 프로필 YAML 파일에서 사용합니다: {cls_yaml_path}")
        else:
            system_template = load_default_system_prompt()
            print(f"Info: 시스템 프롬프트를 기본 YAML 파일에서 사용합니다: {DEFAULT_SYSTEM_PROMPT_PATH}")
        
        # question_profile이 있으면 시스템 프롬프트에 추가
        if question_profile:
            question_profile_text = textwrap.dedent(question_profile).strip()
            system_template = f"{system_template}\n\n**이 과목의 출제 가이드라인:**\n{question_profile_text}"

        # 문항 수 필수 준수 블록: 구조화된 계획으로 '정확히 N문항' 강제
        system_template = (
            f"{system_template}\n\n"
            "**【문항 수 필수 준수】**\n"
            "{quiz_plan_instruction}\n"
            "위 개수·유형을 정확히 지키세요. 1문항이라도 많거나 적게 내면 안 됩니다."
        )

        human_template = (
            "{subject_name} 과목 범위 내에서 퀴즈 문제를 만들어 주세요. "
            "**반드시 아래 출제 계획에 따라 정확히 해당 문항 수만큼만** 출제해주세요. "
            "문제에 대한 **해설도 함께 제공**해주세요. "
            "한국어로 질문을 하면 반드시 대답도 **한국어로 제공**해주세요.\n\n"
            "출제 계획:\n{quiz_plan_instruction}\n\n"
            "사용자 요청 원문: {custom_request}"
        )

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ]
        )

        chain = (
            prompt 
            | llm 
            | JsonOutputParser(pydantic_object=ExamOutput)
        )

        # 필수 필드 검증
        required_fields = ['subject_name']
        for field in required_fields:
            if field not in exam_config or exam_config[field] is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"필수 필드가 누락되었습니다: {field}"
                )

        exam = await chain.ainvoke({
            "persona": persona,
            "context": context,
            "subject_name": exam_config['subject_name'],
            "custom_request": custom_request,
            "format_instructions": format_instructions,
            "quiz_plan_instruction": quiz_plan_instruction,
        })

        # ExamOutput 객체로 변환
        if isinstance(exam, dict):
            exam = ExamOutput(**exam)

        # 에이전트 2단계 이후 백엔드 가드레일:
        # 계획(plan)이 있고 실제 생성 문항 수가 요청 개수보다 많으면
        # 정규식이 아닌 리스트 슬라이스로 개수를 강제로 맞춘다.
        if 'plan' in locals() and isinstance(plan, QuizPlan):
            target_cnt = plan.requested_total
            actual_cnt = len(exam.questions)
            if actual_cnt > target_cnt:
                print(
                    f"Info: 생성 문항 수 초과 감지 — 요청 {target_cnt}문항, 실제 {actual_cnt}문항. "
                    f"앞에서부터 {target_cnt}문항만 사용합니다."
                )
                exam.questions = exam.questions[:target_cnt]

        return {"exam_data": exam}
    except Exception as e:
        print(f"시험지 생성 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating exam: {str(e)}")

async def format_exam_for_printing(exam_data: ExamOutput):
    """시험지 미리보기 용"""
    formatted_exam = []
    formatted_exam.append("=" * 50)
    formatted_exam.append("=" * 50 + "\n")

    for i, q in enumerate(exam_data.questions, 1):
        formatted_exam.append(f"{i}. [{q.item_type_cd}] {q.item_content}")
        if hasattr(q, 'item_choices') and q.item_choices:
            for key, value in q.item_choices.items():
                formatted_exam.append(f"{key}. {value}")
        
        formatted_exam.append("")
        formatted_exam.append(f"정답: {q.item_answer}")
        formatted_exam.append(f"해설: {q.item_explain}")
        formatted_exam.append(f"난이도: {q.item_diff_cd}")
        formatted_exam.append("\n" + "-" * 40 + "\n")

    return "\n".join(formatted_exam)

def generate_item_id(course_id: str) -> str:
    rand_str = ''.join(random.choices(string.ascii_uppercase, k=6))
    return f"ITEM-{course_id}-{rand_str}"


async def save_quiz_to_db(exam_data: ExamOutput, session_id: str, chat_seq: int, course_id: str):
    """퀴즈 데이터를 aita_chatbot_session 테이블의 chat_file_json 컬럼에 저장"""
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        # 퀴즈 데이터를 JSON 형태로 변환
        # 한국 시간대 (UTC+9) 설정
        kst = timezone(timedelta(hours=9))
        kst_time = datetime.now(kst)
        
        # 각 문제에 item_id 추가
        questions_with_id = []
        for question in exam_data.questions:
            question_dict = question.model_dump()
            question_dict["item_id"] = generate_item_id(course_id)
            questions_with_id.append(question_dict)
            
            # ExamQuestion 객체에도 item_id 설정
            question.item_id = question_dict["item_id"]
        
        quiz_json = {
            "quiz_data": {
                "questions": questions_with_id
            },
            "generated_at": kst_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(exam_data.questions)
        }
        
        # 해당 session_id, chat_seq의 chat_file_json 컬럼 업데이트
        await conn.execute('''
            UPDATE aita_chatbot_session 
            SET chat_file_json = $1, upd_dt = CURRENT_TIMESTAMP
            WHERE session_id = $2 AND chat_seq = $3
        ''', json.dumps(quiz_json, ensure_ascii=False), session_id, chat_seq)
        
        await conn.close()
        return True
    except Exception as e:
        print(f"데이터베이스 저장 중 오류 발생: {str(e)}")
        if conn:
            await conn.close()
        raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

async def quizmain(user_id: str, cls_id: str, session_id: str, chat_seq: int, exam_config: dict):
    try:
        # 시험지 생성
        exam_result = await generate_exam(user_id, cls_id, exam_config)
        exam_data = exam_result["exam_data"]
        
        # 교사용 시험지 포맷팅
        formatted_exam = await format_exam_for_printing(exam_data)
        
        # 데이터베이스에 저장 (aita_chatbot_session 테이블의 chat_file_json 컬럼에)
        await save_quiz_to_db(exam_data, session_id, chat_seq, exam_config.get('course_id', cls_id))
        
        return {
            "formatted_exam": formatted_exam,
            "exam_data": exam_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating exam: {str(e)}")

if __name__ == "__main__":
    result = asyncio.run(quizmain(
        #user_id="43819",  # 재료과학개론II 박종민 43819
        #user_id="35033",  #--'무역결제론' 35033 성시일
        #user_id="45932",
        #user_id="20999", #김은주교수
        user_id="12345",
        #cls_id="2024-20-903102-2-01",  # 강의 아이디 창의적코딩 
        #cls_id="2025-20-633004-1-01",  # 무역결제론 
        #cls_id="2025-20-513003-1-01",  # 재료과학개론II 
        #cls_id="2025-20-003039-2-06", #김은주교수 창의코딩-모두의웹
        cls_id="12345",
        session_id="123b4567-3807-4afe-bb5d-df78f7e07ef0",  # 세션 아이디
        chat_seq=0,  # 채팅 시퀀스 번호
        exam_config={
            #'subject_name': "무역결제론",  # 과목명
            #'subject_name': "재료과학개론II", 
            'subject_name': "창의적코딩", 
            #'course_id': "506808",  # 강의 ID "재료과학개론II"
            #'course_id': "633004",  # 강의 ID 무역결제론
            #'course_id': "903102",  # 강의 ID 창의적코딩
            'course_id': "003039", #김은주교수 창의코딩-모두의웹

            'custom_request': "객관식60 문제 생성"
        }
    ))
    
    print("\n=== 시험지 ===")
    print(result["formatted_exam"])
    
    print("\n=== JSON 출력 ===")
    # exam_data를 JSON 형태로 출력
    exam_data_dict = result["exam_data"].to_dict()
    print(json.dumps(exam_data_dict, ensure_ascii=False, indent=2))
