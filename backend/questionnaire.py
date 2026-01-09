# -*- coding: utf-8 -*-
'''
Created on 2024-12-10
@version: 1.1
@author: 고다현
@modified: 고다현 on 2025-02-16
@description: 퀴즈 출제 프로그램 / 소장님이 0216일자로 보내주신 파일을 수정
'''

import os
import json
import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from config import DATABASE_CONFIG

# 환경 변수 설정
# .env 파일에 있는 환경 변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# 기본 모델 정의
class ExamQuestion(BaseModel):
    question: str = Field(description="시험 문제")
    question_type: str = Field(description="문제 유형 (객관식/주관식)")
    options: Optional[List[str]] = Field(description="객관식 문제의 보기 (객관식인 경우에만 필요)", default=None)
    correct_answer: str = Field(description="정답")
    explanation: str = Field(description="문제 해설")
    difficulty: str = Field(description="난이도 (상/중/하)")
    points: int = Field(description="배점")

class ExamOutput(BaseModel):
    questions: List[ExamQuestion]
    total_points: int = Field(description="총점")
    estimated_time: int = Field(description="예상 소요 시간(분)")

    def to_dict(self):
        return {
            "questions": [question.model_dump() for question in self.questions],
            "total_points": self.total_points,
            "estimated_time": self.estimated_time
        }

# Vectorstore 관련 함수들
def get_chroma_vectorstore(cls_id: str):
    """Chroma vectorstore 초기화 및 반환"""
    chroma_db_path = os.path.join("./db/chromadb", cls_id)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embeddings_model,
        collection_name=cls_id
    )
    return vectorstore

async def get_vectorstore(cls_id: str):
    """비동기적으로 vectorstore 가져오기"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_chroma_vectorstore, cls_id)

async def get_relevant_docs(retriever, query: str):
    """관련 문서 검색"""
    return await retriever.ainvoke(query)

def format_docs(docs):
    """문서 포맷팅"""
    return '\n\n'.join([d.page_content for d in docs])

# 기존 코드 중 수정 부분
async def generate_exam(teacher_id: str, class_id: str, exam_config: dict):
    """시험지 생성 함수"""
    vectorstore = await get_vectorstore(class_id)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7
    )

    # 시험 문제 출제 시스템 프롬프트
    system_template = """당신은 전문적인 시험문제 출제자입니다. 
    제공된 교육 자료를 기반으로 학습 목표에 맞는 시험 문제를 생성해야 합니다.
    
    다음 기준을 엄격하게 따라주세요:
    1. 각 문제는 **명확하고 이해하기 쉬워야** 합니다
    2. 난이도 분포: {difficulty_distribution} (상, 중, 하, 예를 들어 상: 20%, 중 50%, 하 30%의 비율로 출제)
    3. 문제 유형: {question_types} 
       - 객관식 문제는 반드시 4개 이상의 보기를 포함해야 합니다
       - 각 보기는 1), 2), 3), 4) 형식으로 제시해야 합니다
       - 정확히 지정된 비율을 따라야 합니다
    4. **총 문제 수는 정확히 {num_questions}개여야 합니다**
    5. 각 문제는 **교육 자료의 내용에 기반**해야 합니다
    6. **총 배점은 100점**이어야 하며, 각 문제의 배점은 **균등하거나** 특정 유형 및 문제의 난이도에 따라 더 많은 배점을 배정할 수 있습니다
    7. 문제지 출제 요청사항: {custom_request}
    
    교육 자료 내용:
    {context}
    
    다음 JSON 형식으로 출력하되, json.loads로 읽을 수 있게 출력해 주세요:
    {{
        "questions": [
            {{
                "question": "문제 내용",
                "question_type": "객관식 또는 주관식",
                "options": ["1) 보기1", "2) 보기2", "3) 보기3", "4) 보기4"], // 객관식인 경우에만
                "correct_answer": "정답",
                "explanation": "해설",
                "difficulty": "난이도(상, 중, 하)",
                "points": 배점
            }}
        ],
        "total_points": 100,
        "estimated_time": 예상 소요시간(분)
    }}
    """

    # human_template = """
    # 위 교육 자료를 바탕으로 시험 문제를 생성해주세요.
    # 각 문제에 대해 문제 내용, 정답, 해설을 포함해주세요.
    # """
    human_template = "강의 자료의 범위내에서 퀴즈 문제를 만들어 주세요. "\
        " 각 문제는 강의에서 다룬 주요 개념을 바탕으로 하되, 문제의 난이도는 **중간 수준**으로 설정하고,"\
        " 객관식 문제는 보기가 4개 이상이어야 합니다."\
        " 다양한 유형의 문제(객관식, 주관식 문제)로 **각 유형별 비율을 설정해 주세요**. "\
        " 예를 들어, 객관식 문제 60%, 주관식 문제 40% 등으로 설정해 주세요. "\
        " 문제에 대한 **해설도 함께 제공**해주세요. 각 문제의 배점은 100점 내에서 **균등하게** 분배하거나, "\
        " 특정 유형 및 문제의 난이도에 따라 더 많은 배점을 배정할 수 있습니다. "\
        " 문제지 출제 요청사항: {custom_request}"

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]
    )

    # 추가 요청사항 처리
    custom_request = exam_config.get('custom_request', '없음')

    # 관련 문서 검색
    docs = await get_relevant_docs(retriever, custom_request)
    context = format_docs(docs)

    chain = (
        prompt 
        | llm 
        | JsonOutputParser(pydantic_object=ExamOutput)
    )

    exam = await chain.ainvoke({
        "context": context,
        "difficulty_distribution": exam_config.get('difficulty_distribution', "상: 20%, 중: 50%, 하: 30%"),
        "question_types": exam_config.get('question_types', "객관식: 60%, 주관식: 30%, 단답형: 10%"),
        "num_questions": exam_config.get('num_questions', 10),
        "custom_request": custom_request
    })

    # ExamOutput 객체로 변환
    if isinstance(exam, dict):
        exam = ExamOutput(**exam)

    # 배점을 조정하여 총점을 100점으로 맞춤
    total_points = sum(q.points for q in exam.questions)
    scaling_factor = 100 / total_points
    for question in exam.questions:
        question.points = round(question.points * scaling_factor)
    exam.total_points = 100

    # 생성된 시험지 저장
    exam_id = await save_exam(teacher_id, class_id, exam, exam_config)
    
    return {"exam_id": exam_id, "exam_data": exam}


async def save_exam(teacher_id: str, class_id: str, exam_data: ExamOutput, exam_config: dict):
    """시험지 데이터베이스 저장"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        exam_id = f"exam-{class_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # ExamOutput 객체를 dict로 변환
        exam_dict = exam_data.model_dump()

        query = """
        INSERT INTO teacher_exams (
            exam_id, teacher_id, cls_id, exam_content, question_content, total_points, 
            estimated_time, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
        """

        await conn.execute(
            query,
            exam_id,
            teacher_id,
            class_id,
            json.dumps(exam_dict, ensure_ascii=False),
            json.dumps(exam_config, ensure_ascii=False),
            exam_dict["total_points"],
            exam_dict["estimated_time"],
            # datetime.now(timezone.utc)
            datetime.utcnow()
        )
        return exam_id
    finally:
        await conn.close()

async def get_exam(exam_id: str):
    """시험지 조회"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        query = """
        SELECT exam_content
        FROM teacher_exams
        WHERE exam_id = $1
        """
        exam_content = await conn.fetchval(query, exam_id)
        
        if not exam_content:
            raise HTTPException(status_code=404, detail="Exam not found")
            
        return ExamOutput(**json.loads(exam_content))
    finally:
        await conn.close()

async def format_exam_for_printing(exam_data: ExamOutput, include_answers: bool = False):
    """시험지 출력 형식 생성"""
    formatted_exam = []
    formatted_exam.append("=" * 50)
    formatted_exam.append(f"총점: {exam_data.total_points}점")
    formatted_exam.append(f"예상 소요시간: {exam_data.estimated_time}분")
    formatted_exam.append("=" * 50 + "\n")

    for i, q in enumerate(exam_data.questions, 1):
        formatted_exam.append(f"{i}. [{q.question_type}] {q.question} ({q.points}점)")
        if hasattr(q, 'options') and q.options:
            formatted_exam.extend(q.options)
        if include_answers:
            formatted_exam.append(f"\n정답: {q.correct_answer}")
            formatted_exam.append(f"해설: {q.explanation}")
            formatted_exam.append(f"난이도: {q.difficulty}")
        formatted_exam.append("\n" + "-" * 40 + "\n")

    return "\n".join(formatted_exam)

async def quizmain(teacher_id: str, cls_id: str, exam_config: dict):
    try:
        # 시험지 생성
        exam_result = await generate_exam(teacher_id, cls_id, exam_config)
        exam_data = exam_result["exam_data"]
        
        # 학생용/교사용 시험지 포맷팅
        student_version = await format_exam_for_printing(exam_data, include_answers=False)
        teacher_version = await format_exam_for_printing(exam_data, include_answers=True)
        
        return {
            "exam_id": exam_result["exam_id"],
            "student_version": student_version,
            "teacher_version": teacher_version,
            "exam_data": exam_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating exam: {str(e)}")

if __name__ == "__main__":
    # 실행 예시
    exam_config = {
        'difficulty_distribution': "상: 20%, 중: 50%, 하: 30%",
        'question_types': "객관식: 70%, 주관식: 30%",
        'num_questions': 10,
        # 'custom_request': "손익계산서와 관련된 내용 위주로 출제해주세요."
        'custom_request': "지도학습에 대한 내용을 출제해주세요."
    }
    
    result = asyncio.run(quizmain(
        teacher_id="teacher1", # 교사 아이디 = prof1
        cls_id="2025-1-511644-01", # 강의 아이디 = C프로그래밍, 자료는 박현제교수님 인공지능 강의자료
        exam_config=exam_config
    ))
    
    print("생성된 시험지 ID:", result["exam_id"])
    print("\n=== 학생용 시험지 ===")
    print(result["student_version"])
    print("\n=== 교사용 시험지 ===")
    print(result["teacher_version"])
# 단답형 추가하고 해당 유형에 대한 설명을 넣어주기 
