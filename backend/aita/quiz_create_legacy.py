'''
과목명 프로파일을 사용하던 초창기~2026.01.08 버전
'''
import os
import json
import asyncio
import yaml
import hashlib
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

def load_subject_config(subject_name: str) -> dict:
    """과목별 설정 정보를 YAML 파일에서 로드"""
    try:
        # 과목명에 따른 YAML 파일명 매핑
        subject_files = {
            # "C++프로그래밍": "cls_cpp_programming.yaml",
            "재료과학개론": "cls_concept_material.yaml",
            "무역결제론": "cls_trade_payment.yaml",
            "창의코딩-모두의인공지능": "cls_creative_coding.yaml"
        }
        
        yaml_filename = subject_files.get(subject_name)
        if not yaml_filename:
            # 기본 설정 반환
            return {
                "subject_characteristics": "기본 교육과정 내용",
                "persona": """당신은 전문적인 교육자이자 시험문제 출제자입니다.
                제공된 교육 자료를 바탕으로 학습 목표에 맞는 고품질의 시험 문제를 생성합니다."""
            }
        
        # YAML 파일 경로 설정
        yaml_path = os.path.join(os.path.dirname(__file__), yaml_filename)
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            subject_data = yaml.safe_load(file)
        
        return {
            "subject_characteristics": subject_data.get('subject_characteristics', ''),
            "persona": subject_data.get('persona', '')
        }
            
    except Exception as e:
        print(f"과목 설정 로드 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본 설정 반환
        return {
            "subject_characteristics": "기본 교육과정 내용",
            "persona": """당신은 전문적인 교육자이자 시험문제 출제자입니다.
            제공된 교육 자료를 바탕으로 학습 목표에 맞는 고품질의 시험 문제를 생성합니다."""
        }


async def generate_exam(user_id: str, class_id: str, exam_config: dict):
    """시험지 생성 함수"""
    try:
        # 과목별 설정 로드
        subject_name = exam_config.get('subject_name', '')
        subject_config = load_subject_config(subject_name)
        subject_characteristics = subject_config['subject_characteristics']
        persona = subject_config['persona']
        
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

        # 시험 문제 출제 시스템 프롬프트
        system_template = """{persona}
        
        제공된 교육 자료를 기반으로 학습 목표에 맞는 시험 문제를 생성해야 합니다.
        
        다음 기준을 엄격하게 따라주세요:
        1. 각 문제는 **명확하고 이해하기 쉬워야** 합니다
        2. **문제 유형과 개수는 사용자의 요청사항에서 유추**해야 합니다:
           - 객관식 문제(MC): 반드시 4개의 보기를 포함하고, 각 보기는 1), 2), 3), 4) 형식으로 제시
             * **객관식 출제 규칙 (필수 준수)**:
               - 각 문항은 정답이 오직 1개만 성립
               - 4지선다 객관식
               - 품질 가드레일:
                 * 단일 정답만 성립하도록 보기 설계(동의어/의미중복 금지)
                 * 오류탐색형은 실제 오류가 존재해야 하며 '원인 또는 위치' 명시
                 * 코드분석형은 출력/호출/수명/순서가 명확한 코드만
                 * 복사/이동/생성자/소멸자 등의 문항은 관련 프로그래밍 언어 동작과 모순 금지
                 * 적어도 2개는 헷갈리는 선지가 있을 수 있도록
                 * (자기검산) 출력에 포함하지 말고 내부적으로 아래를 점검한 뒤 생성:
                   1) 정답이 하나뿐인지
                   2) 오류탐색형에 실제 오류가 있는지
                   3) 관련 프로그래밍 언어 규칙과 일치하는지
                   4) 각 오답지의 '틀린 이유'가 명확한지
           - 빈칸채우기 문제(BLK): 문장에서 핵심 단어나 구문을 빈칸( )으로 만들고 그 빈칸에 들어갈 답을 묻는 문제
           - **사용자가 여러 유형을 요청한 경우 각 유형별로 정확한 개수만큼 출제**해야 합니다
        3. **문제 개수는 사용자 요청사항에서 명시된 개수만큼 정확히 생성**해야 합니다
        4. **난이도는 H(상), M(중), E(하) 중 하나로 설정**해야 합니다
        5. 과목명: {subject_name}
        6. 문제지 출제 요청사항: {custom_request}
           - 이 요청사항에서 문제 유형(객관식, 빈칸채우기, O/X 등)과 문제 개수를 파악하세요
           - 예: "HTML 개요에 관한 객관식 5문제 내줘" → 객관식(MC) 5개 문제
           - 예: "자바스크립트 기초 빈칸채우기 3문제" → 빈칸채우기(BLK) 3개 문제
           - 예: "객관식 3문제 빈칸 4문제 내줘" → 객관식(MC) 3개 + 빈칸채우기(BLK) 4개 문제
           - 예: "O/X 문제 2문제 내줘" → O/X(OX) 2개 문제
        
        교육 자료 내용:
        {context}
        
        {format_instructions}
        """

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
    # 실행 예시
    result = asyncio.run(quizmain(
        user_id="prof1",  # 사용자 아이디
        #cls_id="2024-20-903102-2-01",  # 강의 아이디 창의적코딩 
        #cls_id="2025-20-633004-1-01",  # 무역결제론 
        cls_id="2025-20-513003-1-01",  # 재료과학개론II 
        #cls_id="2025-20-633004-1-01",
        session_id="123b4567-3807-4afe-bb5d-df78f7e07ef0",  # 세션 아이디
        chat_seq=0,  # 채팅 시퀀스 번호
        exam_config={
            #'subject_name': "창의적코딩",  # 과목명
            #'subject_name': "C++프로그래밍",  # 과목명
            #'subject_name': "무역결제론",  # 과목명
            'subject_name': "재료과학개론",  # 과목명
            #'course_id': "903102",  # 강의 ID 창의적코딩
            #'course_id': "633004",  # 강의 ID 무역결제론
            'course_id': "513003",  # 강의 ID 재료과학개론II 
            'custom_request': "재료과학개론 객관식 10문제 난이도 상3, 중4, 하3로 내줘"
            #'custom_request': "C++프로그래밍 중간고사 시험문제를 만들어 줘. 객관식 10문제로 만들어 줘"
            #'custom_request': "클래스와 객체 객관식 5문제"
        }
    ))
    
    print("\n=== 시험지 ===")
    print(result["formatted_exam"])
    
    print("\n=== JSON 출력 ===")
    # exam_data를 JSON 형태로 출력
    exam_data_dict = result["exam_data"].to_dict()
    print(json.dumps(exam_data_dict, ensure_ascii=False, indent=2))
