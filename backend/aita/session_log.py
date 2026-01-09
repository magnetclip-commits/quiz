import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import DATABASE_CONFIG
from llm_factory import get_llm
from datetime import datetime
from typing import List, Dict, Any
from typing import Optional, Dict
import json
import uuid
import asyncpg
from decimal import Decimal
from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging


# 전역 연결 풀 설정
async def init_db_pool():
    return await asyncpg.create_pool(**DATABASE_CONFIG)

# 세션 존재 여부 확인
async def session_exists(pool, student_id: str, class_id: str, session_id: str) -> bool:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 
                FROM aita_chatbot_session 
                WHERE user_id = $1 AND cls_id = $2 AND session_id = $3
            )
            """,
            student_id, class_id, session_id
        )

# 세션 ID 생성 (UUID 기반)
def generate_session_id() -> str:
    """새로운 세션 ID를 생성합니다."""
    return str(uuid.uuid4())

# 사용자 설정 가져오기 (기본값 반환)
async def get_user_prompt_settings(pool, user_id: str) -> dict:
    """사용자 설정을 가져옵니다. 기본값을 반환합니다."""
    return {
        "model_provider": "gpt4o",
        "temperature": 0.7,
        "frequency_penalty": 0.0
    }
# 세션 생성 또는 가져오기
async def get_or_create_session(pool, student_id: str, class_id: str, session_id: str = None) -> str:
    """세션이 없으면 새로 생성하고, 있으면 기존 세션 ID를 반환합니다."""
    async with pool.acquire() as conn:
        try:
            # session_id가 제공되지 않았거나 빈 문자열인 경우 새로 생성
            if not session_id:
                session_id = generate_session_id()
            
            # 기존 세션이 있는지 확인
            result = await conn.fetchval(
                """
                SELECT session_id 
                FROM aita_chatbot_session
                WHERE user_id = $1 AND cls_id = $2 AND session_id = $3
                """, student_id, class_id, session_id
            )
            
            if result:
                return session_id

            # 새 세션의 경우, 실제 질문이 들어올 때 row가 생성됨
            # 여기서는 세션 ID만 반환
            
            return session_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB 오류: {e}")
        
# 질문과 답변을 시퀀스별로 저장 (질문당 1row씩 누적)
async def save_question_and_response(pool, student_id: str, class_id: str, session_id: str, 
                                   question: str, response: str, chat_start_dt: datetime, 
                                   chat_response_dt: datetime, file_title: str = "") -> int:
    """질문과 답변을 시퀀스별로 저장하고 다음 시퀀스 번호를 반환합니다. (질문당 1row씩 누적)"""
    async with pool.acquire() as conn:
        try:
            # 현재 시퀀스 번호 가져오기 (0부터 시작)
            current_seq = await conn.fetchval(
                """
                SELECT COALESCE(MAX(chat_seq), -1) 
                FROM aita_chatbot_session
                WHERE user_id = $1 AND cls_id = $2 AND session_id = $3
                """,
                student_id, class_id, session_id
            )
            
            next_seq = current_seq + 1
            current_time = datetime.now()
            
            # 세션 제목과 세션 시작 시간 가져오기
            session_title = ""
            session_start_time = chat_start_dt  # 기본값은 현재 채팅 시작 시간
            
            if next_seq > 0:
                # 기존 세션 정보 가져오기
                existing_session = await conn.fetchrow(
                    """
                    SELECT session_title, session_start_dt
                    FROM aita_chatbot_session
                    WHERE user_id = $1 AND cls_id = $2 AND session_id = $3
                    LIMIT 1
                    """,
                    student_id, class_id, session_id
                )
                if existing_session:
                    session_title = existing_session['session_title'] or ""
                    session_start_time = existing_session['session_start_dt']
            
            # 새로운 질문/답변 row 삽입
            await conn.execute(
                """
                INSERT INTO aita_chatbot_session (
                    session_id, session_title, user_id, cls_id, session_start_dt,
                    chat_start_dt, chat_response_dt, chat_seq, chat_question, chat_response,
                    chat_file_title, chat_file_json, del_yn, ins_dt, upd_dt
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, '{}', 'N', $12, $13)
                """,
                session_id, session_title, student_id, class_id, 
                session_start_time,  # session_start_dt (첫 번째 질문이면 chat_start_dt, 아니면 기존 세션 시작 시간)
                chat_start_dt, chat_response_dt, next_seq, question, response, file_title,
                current_time, current_time
            )
            
            return next_seq
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"질문/답변 저장 오류: {e}")

# 세션 제목 업데이트
async def update_session_title(pool, student_id: str, class_id: str, session_id: str, 
                              question: str, answer: str):
    """질문과 답변을 기반으로 세션 제목을 자동 생성 후 DB에 업데이트 (모든 row 업데이트)"""
    session_title = await generate_session_title(pool, student_id, question, answer)

    async with pool.acquire() as conn:
        current_time = datetime.now()
        await conn.execute(
            """
            UPDATE aita_chatbot_session
            SET session_title = $1,
                upd_dt = $2
            WHERE user_id = $3 AND cls_id = $4 AND session_id = $5
            """,
            session_title, current_time, student_id, class_id, session_id
        )   

# Decimal 처리를 위한 커스텀 인코더
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# 채팅 기록 로드 함수
async def load_chat_history_from_db(pool, student_id: str, class_id: str, session_id: str) -> List[Dict[str, Any]]:
    """채팅 세션의 모든 질문/답변을 시퀀스 순으로 로드합니다."""
    async with pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT session_id, session_title, user_id, cls_id, session_start_dt, 
                   chat_start_dt, chat_response_dt, chat_seq, chat_question, chat_response, 
                   chat_file_title, chat_file_json, del_yn, ins_dt, upd_dt
            FROM aita_chatbot_session 
            WHERE user_id = $1 AND cls_id = $2 AND session_id = $3
            ORDER BY chat_seq ASC
            """,
            student_id, class_id, session_id
        )
        return [dict(row) for row in results] if results else []

# 간단한 답변 생성 (무조건 "생성하였습니다."로 끝남)
async def generate_simple_response(pool, user_id: str, question: str) -> str:
    """질문에 대한 간단한 답변을 생성합니다. 무조건 '생성하였습니다.'로 끝납니다."""
    user_settings = await get_user_prompt_settings(pool, user_id)
    llm = get_llm(user_settings)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 교육용 AI 어시스턴트입니다. 사용자의 요청에 대해 간단한 확인 메시지만 제공하세요. 실제 문제나 내용을 생성하지 말고, 단순히 '~에 대한 ~를 생성하였습니다.' 형태의 간단한 메시지만 작성하세요. 답변은 반드시 '생성하였습니다.'로 끝나야 합니다."),
        ("human", "다음 요청에 대해 간단한 확인 메시지만 작성해주세요 (예를 들면 문제 생성을 요청했을때 실제 문제 생성하지 말고, '~에 대한 문제를 생성하였습니다.' 형태로만):\n\n{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = await chain.ainvoke({
        "question": question
    })
    
    # "생성하였습니다."로 끝나지 않으면 강제로 추가
    response = response.strip()
    if not response.endswith("생성하였습니다."):
        response += " 생성하였습니다."
    
    return response

# 파일명 생성
async def generate_file_title(pool, user_id: str, question: str) -> str:
    """질문에 기반하여 파일명을 생성합니다."""
    user_settings = await get_user_prompt_settings(pool, user_id)
    llm = get_llm(user_settings)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 파일명 생성 전문가입니다. 사용자의 질문을 기반으로 적절한 파일명을 생성해주세요. 파일명은 한글로 30자 이내로 작성하고, 언더스코어(_)나 하이픈(-)을 사용하여 구분해주세요. 확장자는 포함하지 마세요."),
        ("human", "다음 질문에 대한 적절한 파일명을 생성해주세요:\n\n{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    file_title = await chain.ainvoke({
        "question": question
    })
    
    # 파일명 정리 (특수문자 제거, 길이 제한)
    file_title = file_title.strip().strip('"').strip("'")
    # 파일명에 사용할 수 없는 문자 제거
    import re
    file_title = re.sub(r'[<>:"/\\|?*]', '', file_title)
    # 길이 제한 (30자)
    if len(file_title) > 30:
        file_title = file_title[:30]
    
    return file_title

# 세션 제목 생성
async def generate_session_title(pool, user_id: str, question: str, answer: str) -> str:
    """질문과 답변을 기반으로 세션 제목을 생성합니다."""
    user_settings = await get_user_prompt_settings(pool, user_id)
    llm = get_llm(user_settings)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 대화 내용을 간단하고 명확한 제목으로 요약하는 전문가입니다. 제목은 한글로 20자 이내로 작성해주세요."),
        ("human", "다음 대화 내용을 세션 제목으로 요약해주세요:\n\n질문: {question}\n답변: {answer}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    title = await chain.ainvoke({
        "question": question,
        "answer": answer
    })
    return title.strip().strip('"').strip("'")

# 메인 처리 함수
async def process_chat_session(user_id: str, class_id: str, session_id: str, question: str) -> dict:
    """채팅 세션을 처리하는 메인 함수"""
    pool = await init_db_pool()
    
    try:
        # 1. 세션 생성 또는 가져오기
        final_session_id = await get_or_create_session(pool, user_id, class_id, session_id)
        print(f"세션 ID: {final_session_id}")
        
        # 2. 채팅 시작 시간 기록
        chat_start_dt = datetime.now()
        print(f"채팅 시작 시간: {chat_start_dt}")
        
        # 3. 간단한 답변 생성
        simple_response = await generate_simple_response(pool, user_id, question)
        print(f"생성된 답변: {simple_response}")
        
        # 4. 파일명 생성
        file_title = await generate_file_title(pool, user_id, question)
        print(f"생성된 파일명: {file_title}")
        
        # 5. 채팅 응답 완료 시간 기록
        chat_response_dt = datetime.now()
        print(f"채팅 응답 완료 시간: {chat_response_dt}")
        
        # 6. 질문과 답변을 시퀀스별로 저장
        seq_number = await save_question_and_response(
            pool, user_id, class_id, final_session_id, question, simple_response, 
            chat_start_dt, chat_response_dt, file_title
        )
        print(f"저장된 시퀀스 번호: {seq_number}")
        
        # 6. 세션 제목 업데이트 (첫 번째 질문인 경우에만)
        if seq_number == 0:
            await update_session_title(pool, user_id, class_id, final_session_id, question, simple_response)
            print("세션 제목이 업데이트되었습니다.")
        
        return {
            "session_id": final_session_id,
            "seq": seq_number,
            "question": question,
            "response": simple_response,
            "file_title": file_title,
            "chat_start_dt": chat_start_dt.isoformat(),
            "chat_response_dt": chat_response_dt.isoformat(),
            "processing_time": (chat_response_dt - chat_start_dt).total_seconds(),
            "status": "success"
        }
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return {
            "session_id": session_id,
            "error": str(e),
            "status": "error"
        }
    finally:
        await pool.close()

if __name__ == "__main__":
    import asyncio
    
    # 변수 설정
    user_id = "prof1"
    cls_id = "2025-20-506808-1-01"
    #session_id = None  # 새 세션 생성
    session_id = "123b4567-3807-4afe-bb5d-df78f7e07ef0"
    custom_request = "포인터에 관한 빈칸 3문제 내줘"
    
    # 채팅 세션 처리 실행
    result = asyncio.run(process_chat_session(user_id, cls_id, session_id, custom_request))
    print(f"결과: {result}")
