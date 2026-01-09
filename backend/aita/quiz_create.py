'''
2025.12.29 /hlta/backend/aita/yaml 의 prompt 파일 사용 
2026.01.06 question_profile 추가, OpenAIEmbeddings(model="text-embedding-3-large" 추가
2026.01.07 yaml/ 이하 폴더명 수정(CLS, USR, SYS)
'''
import os
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
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from tutor.config import DATABASE_CONFIG
from config import DATABASE_CONFIG
import random
import string

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
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치해주세요.")

# 기본 모델 정의
class ExamQuestion(BaseModel):
    item_content: str = Field(description="시험 문제")
    item_type_cd: str = Field(description="문제 유형 코드 (MC: 객관식, BLK: 빈칸채우기)")
    # item_type_cd: str = Field(description="문제 유형 코드 (MC: 객관식, OX: OX문제, SA: 단답형, ESS: 서술형, BLK: 빈칸채우기)")
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

# Vectorstore 관련 함수들
def get_chroma_vectorstore(cls_id: str):
    """Chroma vectorstore 초기화 및 반환"""
    try:
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
        # 사용자별 설정 로드
        user_config = load_user_config(user_id)
        persona = user_config['persona']
        
        # 클래스별 설정 로드
        cls_config = load_cls_config(class_id)
        subject_name = cls_config.get('subject_name') or exam_config.get('subject_name', '')
        subject_characteristics = cls_config['subject_characteristics']
        question_profile = cls_config.get('question_profile', '')
        system_prompt_template = cls_config.get('system_prompt')
        
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
                print(f"Info: {class_id} 강의에서 '{custom_request}'와 관련된 문서를 찾을 수 없어 과목 특성 기반으로 시험지를 생성합니다.")

        llm = ChatOpenAI(
            # model="gpt-4o",
            model="gpt-5.2",
            temperature=0.2,
            api_key=OPENAI_API_KEY
        )

        # JsonOutputParser를 사용한 형식 지시사항 생성
        output_parser = JsonOutputParser(pydantic_object=ExamOutput)
        format_instructions = output_parser.get_format_instructions()

        # 시험 문제 출제 시스템 프롬프트 (yaml에서 가져오거나 기본값 사용)
        if system_prompt_template:
            system_template = textwrap.dedent(system_prompt_template).strip()
        else:
            system_template = load_default_system_prompt()
        
        # question_profile이 있으면 시스템 프롬프트에 추가
        if question_profile:
            question_profile_text = textwrap.dedent(question_profile).strip()
            system_template = f"{system_template}\n\n**이 과목의 출제 가이드라인:**\n{question_profile_text}"

        human_template = "{subject_name} 과목 범위 내에서 퀴즈 문제를 만들어 주세요. "\
            " 사용자의 요청사항에서 문제 유형과 개수를 파악하여 출제해주세요."\
            " 문제에 대한 **해설도 함께 제공**해주세요. "\
            " 한국어로 질문을 하면 반드시 대답도 **한국어로 제공**해주세요."\
            " 문제지 출제 요청사항: {custom_request}"

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ]
        )

        # 추가 요청사항 처리
        custom_request = exam_config.get('custom_request', '없음')

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
            "format_instructions": format_instructions
        })

        # ExamOutput 객체로 변환
        if isinstance(exam, dict):
            exam = ExamOutput(**exam)

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
        user_id="20999", #김은주교수
        #cls_id="2024-20-903102-2-01",  # 강의 아이디 창의적코딩 
        #cls_id="2025-20-633004-1-01",  # 무역결제론 
        #cls_id="2025-20-513003-1-01",  # 재료과학개론II 
        cls_id="2025-20-003039-2-06", #김은주교수 창의코딩-모두의웹
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

            'custom_request': "창의적코딩 객관식 10문제 난이도 상3, 중4, 하3로 내줘"
        }
    ))
    
    print("\n=== 시험지 ===")
    print(result["formatted_exam"])
    
    print("\n=== JSON 출력 ===")
    # exam_data를 JSON 형태로 출력
    exam_data_dict = result["exam_data"].to_dict()
    print(json.dumps(exam_data_dict, ensure_ascii=False, indent=2))
