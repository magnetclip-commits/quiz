"""
강의계획서(aici_plan, aici_plan_week),강의자료 데이터를 바탕으로 과목 특성과 출제 프로파일을 생성해 YAML로 저장하는 스크립트.
2026.01.07: DB정보가 없으면 에러생성하지 말고 강의자료 바탕으로 프로파일 생성 
2026.01.08: db에 저장 후 yaml로 저장하는 방식으로 변경
사용법:
  python cls_profile_generator.py 2025-20-513003-1-01
"""

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from io import StringIO
import asyncpg
import oracledb
import chromadb
import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

# 프로젝트 루트의 config 모듈을 찾기 위한 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import DATABASE2_CONFIG
from config import DATABASE_CONFIG  # PostgreSQL 

BASE_DIR = Path(__file__).resolve().parent
YAML_DIR = BASE_DIR / "yaml" / "CLS"

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치해주세요.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST", "hlta-chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


class ClassProfile(BaseModel):
    """LLM이 반환해야 하는 프로파일 스키마"""

    subject_name: str = Field(description="강의명")
    subject_characteristics: str = Field(description="강의 특성, 핵심 토픽, 제한사항 등 서술")
    question_profile: str = Field(description="이 과목에서 문제를 낼 때의 규칙/스타일/주의사항")


def _parse_cls_id(cls_id: str):
    parts = cls_id.split("-")
    if len(parts) < 5:
        raise ValueError(f"cls_id 형식이 올바르지 않습니다: {cls_id}")
    plan_yy = parts[0]
    plan_hakgi = parts[1][0]
    plan_bookcode = parts[2]
    plan_bunban = parts[4]
    return plan_yy, plan_hakgi, plan_bookcode, plan_bunban


def fetch_plan_summary_sync(conn: oracledb.Connection, cls_id: str) -> dict:
    """AICI_PLAN의 한 행을 cls_id를 파싱해 조회"""
    plan_yy, plan_hakgi, plan_bookcode, plan_bunban = _parse_cls_id(cls_id)

    query = """
    SELECT *
    FROM HUIS.AICI_PLAN
    WHERE PLAN_YY = :PLAN_YY
      AND PLAN_HAKGI = :PLAN_HAKGI
      AND PLAN_BOOKCODE = :PLAN_BOOKCODE
      AND PLAN_BUNBAN = :PLAN_BUNBAN
    FETCH FIRST 1 ROWS ONLY
    """
    cursor = conn.cursor()
    cursor.execute(
        query,
        {
            "PLAN_YY": plan_yy,
            "PLAN_HAKGI": plan_hakgi,
            "PLAN_BOOKCODE": plan_bookcode,
            "PLAN_BUNBAN": plan_bunban,
        },
    )
    row = cursor.fetchone()
    columns = [col[0].lower() for col in cursor.description] if cursor.description else []
    cursor.close()
    if not row:
        return {}
    return {columns[i]: row[i] for i in range(len(columns))}


async def fetch_plan_summary(conn: oracledb.Connection, cls_id: str) -> dict:
    return await asyncio.to_thread(fetch_plan_summary_sync, conn, cls_id)


def fetch_weekly_plan_sync(conn: oracledb.Connection, cls_id: str):
    """AICI_PLAN_WEEK 주차별 데이터 배열 조회"""
    plan_yy, plan_hakgi, plan_bookcode, plan_bunban = _parse_cls_id(cls_id)

    query = """
    SELECT
        PLANW_WEEK,
        PLANW_GOAL,
        PLANW_LECTURE,
        PLANW_METHOD,
        PLANW_BIGO
    FROM HUIS.AICI_PLAN_WEEK
    WHERE PLANW_YY = :PLANW_YY
      AND PLANW_HAKGI = :PLAN_HAKGI
      AND PLANW_BOOKCODE = :PLAN_BOOKCODE
      AND PLANW_BUNBAN = :PLAN_BUNBAN
    ORDER BY TO_NUMBER(REGEXP_REPLACE(PLANW_WEEK, '[^0-9]', ''))
    """
    cursor = conn.cursor()
    cursor.execute(
        query,
        {
            "PLANW_YY": plan_yy,
            "PLAN_HAKGI": plan_hakgi,
            "PLAN_BOOKCODE": plan_bookcode,
            "PLAN_BUNBAN": plan_bunban,
        },
    )
    columns = [col[0].lower() for col in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    cursor.close()
    return [{columns[i]: row[i] for i in range(len(columns))} for row in rows] if rows else []


async def fetch_weekly_plan(conn: oracledb.Connection, cls_id: str):
    return await asyncio.to_thread(fetch_weekly_plan_sync, conn, cls_id)


def fetch_class_name_sync(conn: oracledb.Connection, cls_id: str) -> str:
    """AICI_PLAN의 PLAN_BOOKCODE_NM으로 강의명 조회 (없으면 빈 문자열)"""
    plan_yy, plan_hakgi, plan_bookcode, plan_bunban = _parse_cls_id(cls_id)

    query = """
    SELECT PLAN_BOOKCODE_NM
    FROM HUIS.AICI_PLAN
    WHERE PLAN_YY = :PLAN_YY
      AND PLAN_HAKGI = :PLAN_HAKGI
      AND PLAN_BOOKCODE = :PLAN_BOOKCODE
      AND PLAN_BUNBAN = :PLAN_BUNBAN
    FETCH FIRST 1 ROWS ONLY
    """
    cursor = conn.cursor()
    cursor.execute(
        query,
        {
            "PLAN_YY": plan_yy,
            "PLAN_HAKGI": plan_hakgi,
            "PLAN_BOOKCODE": plan_bookcode,
            "PLAN_BUNBAN": plan_bunban,
        },
    )
    row = cursor.fetchone()
    cursor.close()
    if not row:
        return ""
    return row[0] or ""


async def fetch_class_name(conn: oracledb.Connection, cls_id: str) -> str:
    return await asyncio.to_thread(fetch_class_name_sync, conn, cls_id)


def get_chroma_vectorstore(cls_id: str):
    """Chroma vectorstore 초기화 및 반환"""
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            client=CHROMA_CLIENT,
            embedding_function=embeddings_model,
            collection_name=cls_id
        )
        return vectorstore
    except Exception as e:
        print(f"ChromaDB 초기화 실패: {str(e)}")
        return None


async def fetch_chroma_documents(cls_id: str, max_docs: int = 20) -> list:
    """ChromaDB에서 강의 자료 검색 (과목 개요, 핵심 내용 등)"""
    try:
        vectorstore = get_chroma_vectorstore(cls_id)
        if vectorstore is None:
            return []
        
        # 컬렉션 존재 여부 확인
        try:
            collection = vectorstore._collection
            doc_count = collection.count()
            if doc_count == 0:
                print(f"경고: ChromaDB에 {cls_id} 컬렉션은 존재하지만 문서가 없습니다.")
                return []
        except Exception as e:
            print(f"경고: ChromaDB 컬렉션 확인 실패 ({cls_id}): {str(e)}")
            return []
        
        # 과목 개요 및 핵심 내용을 검색하기 위한 쿼리들
        queries = [
            "과목 개요 강의 소개 학습 목표",
            "핵심 개념 주요 내용",
            "강의 계획 수업 내용"
        ]
        
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda q: vectorstore.similarity_search(q, k=max_docs // len(queries) + 1),
                    query
                )
                
                for doc in docs:
                    # 중복 제거 (내용이 비슷한 문서 제외)
                    content_hash = hash(doc.page_content[:200])  # 앞 200자로 해시
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                        
                        if len(all_docs) >= max_docs:
                            break
                            
                if len(all_docs) >= max_docs:
                    break
            except Exception as e:
                print(f"ChromaDB 검색 중 오류 (쿼리: {query}): {str(e)}")
                continue
        
        print(f"ChromaDB에서 {len(all_docs)}개 문서를 검색했습니다.")
        return all_docs[:max_docs]
        
    except Exception as e:
        print(f"ChromaDB 문서 검색 실패: {str(e)}")
        return []


def build_prompt(plan_meta: dict, weekly_plan: list, chroma_docs: list = None) -> ChatPromptTemplate:
    """LLM 입력 프롬프트 생성"""
    parser = JsonOutputParser(pydantic_object=ClassProfile)

    system_tmpl = """너는 대학 강의계획서를 분석해 과목 특성과 출제 전략을 만드는 전문가다.
입력으로 주어지는 강의 메타데이터와 주차별 계획을 정리해라.
결과는 한국어로 작성하고, 제공된 출력 스키마를 반드시 준수한다."""

    # ChromaDB 문서가 있으면 추가
    chroma_section = ""
    if chroma_docs and len(chroma_docs) > 0:
        chroma_contents = []
        for i, doc in enumerate(chroma_docs[:10], 1):  # 최대 10개만 사용
            title = doc.get("metadata", {}).get("title", f"문서 {i}")
            content = doc.get("content", "")[:1000]  # 각 문서는 최대 1000자만
            chroma_contents.append(f"[문서 {i}: {title}]\n{content}")
        
        chroma_section = f"""
3) 강의 자료 내용(ChromaDB에서 검색된 문서): 
{chr(10).join(chroma_contents)}

**주의: ChromaDB 문서는 강의의 실제 내용을 보완하는 참고 자료입니다. 강의계획서와 충돌하는 경우 강의계획서를 우선하되, ChromaDB 문서의 구체적인 내용을 활용해 프로파일을 더 풍부하게 작성하세요.**
"""

    # human_tmpl은 일반 문자열로 만들고 chroma_section의 중괄호를 이스케이프
    # chroma_section의 모든 중괄호를 이중 중괄호로 변환하여 템플릿 변수와 충돌 방지
    escaped_chroma_section = chroma_section.replace("{", "{{").replace("}", "}}")
    
    # f-string을 사용하지 않고 문자열 연결 사용
    human_tmpl = """
아래 JSON을 참고해 과목 특성과 출제 프로파일을 작성해줘.
1) 강의 메타데이터(comment_keyed_row): {plan_meta}
2) 주차별 계획(result_json): {weekly_plan}""" + escaped_chroma_section + """

요구사항:
- 'subject_characteristics'에는 **학습 내용 중심**으로 작성:
  * 과목 목표, 핵심 토픽, 주요 개념, 이론/실습 내용
  * 난이도/선수지식, 제외할 범위
  * **주의: 강의 운영 방식(강의식 비율, 토의 비율, 평가 방법 등)은 포함하지 말 것**
  * **주의: 수업 진행 방식이나 운영 특성에 대한 내용은 제외하고, 실제 학습해야 할 지식/기술/개념만 포함**
- 'question_profile'에는 문제 유형/난이도 배분, 금지/권장 사항, 정답 유일성/오답 설계 원칙, 해설 톤을 포함
- 실제 수업 용어/주차 목표를 최대한 반영
- 불확실한 정보는 추정 대신 '제공되지 않음'으로 명시
- ChromaDB 문서가 제공된 경우, 해당 문서의 구체적인 내용을 활용해 더 정확하고 상세한 프로파일을 작성하세요

{format_instructions}
"""

    return ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_tmpl),
            HumanMessagePromptTemplate.from_template(human_tmpl),
        ]
    ), parser


async def generate_profile(plan_meta: dict, weekly_plan: list, chroma_docs: list = None) -> ClassProfile:
    """LLM 호출로 프로파일 생성"""
    prompt, parser = build_prompt(plan_meta, weekly_plan, chroma_docs)
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    chain = prompt | llm | parser
    result = await chain.ainvoke(
        {
            "plan_meta": json.dumps(plan_meta, ensure_ascii=False),
            "weekly_plan": json.dumps(weekly_plan, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
        }
    )
    # parser 결과가 dict로 반환될 수 있으므로 모델로 보정
    if isinstance(result, ClassProfile):
        return result
    return ClassProfile.model_validate(result)


def profile_to_yaml_text(profile: ClassProfile) -> str:
    """생성된 프로파일을 YAML 텍스트로 변환"""
    def _normalize(text: str) -> str:
        """텍스트 정규화: 앞뒤 공백 제거, 들여쓰기 정리"""
        if not text:
            return ""
        # 들여쓰기 제거 및 트리밍
        normalized = textwrap.dedent(text).strip()
        return normalized

    def _format_multiline_text(text: str) -> str:
        """멀티라인 텍스트를 YAML literal 블록 형식으로 포맷팅"""
        normalized = _normalize(text)
        # 줄바꿈 정리: 연속된 빈 줄은 하나로, 끝의 빈 줄 제거
        lines = normalized.split("\n")
        # 연속된 빈 줄 제거
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                cleaned_lines.append(line)
            prev_empty = is_empty
        
        # 끝의 빈 줄 제거
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return "\n".join(cleaned_lines)

    subject_name = _normalize(profile.subject_name)
    subject_characteristics = _format_multiline_text(profile.subject_characteristics)
    question_profile = _format_multiline_text(profile.question_profile)

    # YAML 파일 내용을 문자열로 생성
    output = StringIO()
    
    # subject_name 저장
    if "\n" in subject_name or len(subject_name) > 80:
        output.write("subject_name: |\n")
        for line in subject_name.split("\n"):
            output.write(f"  {line}\n")
    else:
        output.write(f'subject_name: "{subject_name}"\n')
    
    output.write("\n")
    
    # subject_characteristics 저장
    output.write("subject_characteristics: |\n")
    for line in subject_characteristics.split("\n"):
        output.write(f"  {line}\n")
    
    output.write("\n")
    
    # question_profile 저장
    output.write("question_profile: |\n")
    for line in question_profile.split("\n"):
        output.write(f"  {line}\n")
    
    return output.getvalue()


def save_profile_to_yaml(cls_id: str, profile: ClassProfile):
    """생성된 프로파일을 cls_id.yaml로 저장 (기존 있으면 덮어씀)
    원자적 쓰기를 위해 임시 파일에 먼저 쓰고 원본 파일로 rename
    """
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    output_path = YAML_DIR / f"{cls_id}.yaml"
    temp_path = YAML_DIR / f"{cls_id}.yaml.tmp"
    
    yaml_text = profile_to_yaml_text(profile)
    
    # 임시 파일에 먼저 쓰기
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    
    # 원자적으로 원본 파일로 rename (임시 파일이 완전히 쓰여진 후)
    temp_path.replace(output_path)

    return output_path


async def save_profile_to_db(emp_no: str, cls_id: str, yaml_text: str) -> None:
    """생성된 프로파일을 데이터베이스에 저장"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        file_path = f"/profiles/CLS/{cls_id}.yaml"
        await conn.execute(
            "CALL sp_aita_profile_save($1, $2, $3, $4, $5)",
            emp_no,  # 첫 번째 파라미터: plan_gyosu_id (emp_no)
            cls_id,  # 두 번째 파라미터: cls_id
            'CLS',   # 세 번째 파라미터: 타입
            yaml_text,  # 네 번째 파라미터: YAML 텍스트
            file_path   # 다섯 번째 파라미터: 파일 경로
        )
        print(f"DB 저장 완료: {cls_id} (emp_no: {emp_no})")
    finally:
        await conn.close()


async def run(cls_id: str):
    """전체 파이프라인 실행"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

    if not DATABASE2_CONFIG.get("dsn"):
        raise RuntimeError("DATABASE2_DSN 환경 변수가 설정되어 있지 않습니다.")

    conn = oracledb.connect(
        user=DATABASE2_CONFIG.get("user"),
        password=DATABASE2_CONFIG.get("password"),
        dsn=DATABASE2_CONFIG.get("dsn"),
    )
    try:
        plan_meta = await fetch_plan_summary(conn, cls_id)
        weekly_plan = await fetch_weekly_plan(conn, cls_id)
        class_name = await fetch_class_name(conn, cls_id)

        if not plan_meta:
            print(f"경고: {cls_id} 에 대한 aici_plan 데이터가 없습니다. 사용 가능한 정보만으로 프로파일을 생성합니다.")
            plan_meta = {"meta": "제공되지 않음"}

        # ChromaDB에서 강의 자료 검색
        print(f"ChromaDB에서 {cls_id} 강의 자료를 검색 중...")
        chroma_docs = await fetch_chroma_documents(cls_id, max_docs=20)
        
        if chroma_docs:
            print(f"ChromaDB에서 {len(chroma_docs)}개 문서를 찾았습니다.")
        else:
            print(f"ChromaDB에 {cls_id} 관련 문서가 없거나 검색에 실패했습니다.")

        profile = await generate_profile(plan_meta, weekly_plan, chroma_docs)
        if class_name:
            profile.subject_name = class_name
        elif not (profile.subject_name or "").strip():
            profile.subject_name = cls_id
        
        # YAML 텍스트 생성
        yaml_text = profile_to_yaml_text(profile)
        
        # plan_gyosu_id 추출 (emp_no로 사용)
        emp_no = plan_meta.get('plan_gyosu_id') if plan_meta else None
        if not emp_no:
            # plan_gyosu_id가 없으면 빈 문자열이나 cls_id를 사용할 수도 있지만,
            # 일반적으로는 필수 값이므로 경고 메시지 출력
            print(f"경고: {cls_id}에 대한 plan_gyosu_id가 없습니다. cls_id를 사용합니다.")
            emp_no = cls_id
        
        # 먼저 DB에 저장
        await save_profile_to_db(str(emp_no), cls_id, yaml_text)
        
        # 그 다음 YAML 파일로 저장
        output_path = save_profile_to_yaml(cls_id, profile)
        print(f"YAML 파일 생성 완료: {output_path}")
    finally:
        conn.close()


def main(cls_id: str):
    """cls_id를 인자로 받아 실행"""
    asyncio.run(run(cls_id))


if __name__ == "__main__":
    # 인자를 주면 인자 값을 사용, 인자가 없으면 하드코딩된 기본값 사용
    if len(sys.argv) >= 2:
        cls_id = sys.argv[1]
    else:
        #cls_id = "2025-20-513003-1-01" # 재료과학개론II 
        cls_id = "2025-20-633004-1-01"  # 무역결제론 
        #cls_id = "2024-20-903102-2-01"  # 창의적코딩 

    main(cls_id)