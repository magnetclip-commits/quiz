"""
교수자 정보(HUIS.RESEARCHER_MATCHING_USER_INFO 등) 데이터를 바탕으로
교수자 프로필을 생성해 YAML로 저장하는 스크립트.
2026.01.08 db에 저장 후 yaml로 저장하는 방식으로 변경
사용법:
  python prof_profile_generator.py 35033
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
import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 프로젝트 루트의 config 모듈을 찾기 위한 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import DATABASE2_CONFIG  # 한림대 오라클
from config import DATABASE_CONFIG  # PostgreSQL

BASE_DIR = Path(__file__).resolve().parent
YAML_DIR = BASE_DIR / "yaml" / "USR"

# .env 파일 로드
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치해주세요.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ProfessorProfile(BaseModel):
    """LLM이 반환해야 하는 프로파일 스키마"""

    name: str = Field(description="교수자 이름 (국문 또는 영문)")
    persona: str = Field(description="교수자의 전문 분야, 연구 관심사, 출제 스타일 등을 포함한 상세 프로필")


def fetch_professor_info_sync(conn: oracledb.Connection, emp_no: str) -> dict:
    """교수자 정보 조회 (오라클 DB 쿼리)"""
    query = """
    WITH U AS (
        SELECT
            EMPLOYEE_NUMBER,
            KOR_NAME,
            ENG_NAME,
            INSTITUTION_NAME,
            DEPT_NAME,
            POSITION_NAME,
            RESPONSIBILITY_OFFICE_NAME,
            ENTER_DATE
        FROM HUIS.RESEARCHER_MATCHING_USER_INFO
        WHERE EMPLOYEE_NUMBER = :EMP_NO
    ),
    F AS (
        SELECT
            EMPLOYEE_NUMBER,
            LISTAGG(DESCRIPTION, ' | ')
                WITHIN GROUP (ORDER BY SEQ_NO) AS RESEARCH_FIELDS
        FROM HUIS.RESEARCHER_MATCHING_PROF_RESEARCH_FIELD
        WHERE EMPLOYEE_NUMBER = :EMP_NO
        GROUP BY EMPLOYEE_NUMBER
    ),
    T0 AS (
        SELECT
            EMPLOYEE_NUMBER,
            PUBLICATION_DATE,
            ORIGINAL_THESIS,
            ANOTHER_THESIS,
            JOURNAL_NAME,
            JOURNAL_DIVISION_NAME,
            LANGUAGE_DIVISION,
            NATION_DIVISION,
            ROW_NUMBER() OVER (
                PARTITION BY EMPLOYEE_NUMBER
                ORDER BY PUBLICATION_DATE DESC NULLS LAST
            ) AS RN
        FROM HUIS.RESEARCHER_MATCHING_THESIS_INFO
        WHERE EMPLOYEE_NUMBER = :EMP_NO
    ),
    T AS (
        -- RECENT THESIS TOP 5 PREVIEW
        SELECT
            EMPLOYEE_NUMBER,
            LISTAGG(
                '[' || NVL(PUBLICATION_DATE, '-') || '] '
                || NVL(ORIGINAL_THESIS, NVL(ANOTHER_THESIS, '(제목없음)'))
                || ' / ' || NVL(JOURNAL_NAME, '-')
                || ' / ' || NVL(JOURNAL_DIVISION_NAME, '-')
                || ' / ' || NVL(LANGUAGE_DIVISION, '-')
                || ' / ' || NVL(NATION_DIVISION, '-'),
                CHR(10)
            ) WITHIN GROUP (ORDER BY PUBLICATION_DATE DESC NULLS LAST)
            AS RECENT_THESIS_TOPN
        FROM T0
        WHERE RN <= 5
        GROUP BY EMPLOYEE_NUMBER
    )
    SELECT
        U.EMPLOYEE_NUMBER            AS 교수사번,
        U.KOR_NAME                   AS 교수명_국문,
        U.ENG_NAME                   AS 교수명_영문,
        U.INSTITUTION_NAME           AS 소속기관,
        U.DEPT_NAME                  AS 소속학과,
        U.POSITION_NAME              AS 직위,
        U.RESPONSIBILITY_OFFICE_NAME AS 보직,
        U.ENTER_DATE                 AS 임용일,
        F.RESEARCH_FIELDS            AS 연구분야_리스트,
        T.RECENT_THESIS_TOPN         AS 최근논문_TOP5_미리보기
    FROM U
    LEFT JOIN F ON F.EMPLOYEE_NUMBER = U.EMPLOYEE_NUMBER
    LEFT JOIN T ON T.EMPLOYEE_NUMBER = U.EMPLOYEE_NUMBER
    """

    cursor = conn.cursor()
    cursor.execute(query, {"EMP_NO": emp_no})
    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()

    if not row:
        return {}

    # 결과를 딕셔너리로 변환
    return {columns[i]: row[i] for i in range(len(columns))}


async def fetch_professor_info(conn: oracledb.Connection, emp_no: str) -> dict:
    """교수자 정보 조회 (비동기 래퍼)"""
    return await asyncio.to_thread(fetch_professor_info_sync, conn, emp_no)


def build_prompt(prof_info: dict):
    """LLM 입력 프롬프트 생성"""
    parser = JsonOutputParser(pydantic_object=ProfessorProfile)

    system_tmpl = """너는 대학 교수자의 연구 배경과 전문성을 분석해 교수자 프로필을 만드는 전문가다.
    입력으로 주어지는 교수자의 기본 정보, 연구 분야, 최근 논문 정보를 바탕으로 상세한 프로필을 작성해라.

    중요 규칙:
    - 입력 데이터에 없는 내용은 절대 추정하거나 보완하지 말 것
    - '제공되지 않음', '확인 불가', '추정', '알 수 없음'과 같은 표현을 사용하지 말 것
    - 정보가 없는 항목은 섹션/목차/불릿 자체를 만들지 말고 완전히 생략할 것

    결과는 한국어로 작성하고, 제공된 출력 스키마를 반드시 준수한다.
    """


    human_tmpl = """
    아래 교수자 정보를 참고해 교수자 프로필을 작성해줘.

    교수자 정보:
    {prof_info}

    요구사항: 
    - 'name'에는 교수자 이름을 국문 또는 영문으로 작성 
    - 'persona'에는 다음 내용을 포함: 
    1. 교수자의 전문 분야 및 연구 관심사 (연구분야_리스트를 참고) 
    2. 최근 연구 활동 (최근논문_TOP5_미리보기를 참고하여 연구 주제와 경향 파악) 
    3. 소속 및 직위 정보 
    4. 전문 용어 및 핵심 개념 (해당 분야의 주요 용어와 개념 나열) 
    - 실제 논문 제목과 연구 분야를 최대한 반영 
    - 불확실한 정보는 추정 대신 작성하지 않음
    - 프로필은 자연스러운 문장으로 작성하되, 구조화된 bullet point 형식도 활용 가능

    {format_instructions}
    """


    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_tmpl),
            HumanMessagePromptTemplate.from_template(human_tmpl),
        ]
    )
    return prompt, parser


async def generate_profile(prof_info: dict) -> ProfessorProfile:
    """LLM 호출로 프로파일 생성"""
    prompt, parser = build_prompt(prof_info)

    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

    chain = prompt | llm | parser
    result = await chain.ainvoke(
        {
            "prof_info": json.dumps(prof_info, ensure_ascii=False, default=str),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # parser 결과가 dict로 반환될 수 있으므로 모델로 보정
    if isinstance(result, ProfessorProfile):
        return result
    return ProfessorProfile.model_validate(result)


def profile_to_yaml_text(profile: ProfessorProfile) -> str:
    """생성된 프로파일을 YAML 텍스트로 변환"""
    class _LiteralDumper(yaml.SafeDumper):
        """멀티라인 문자열을 YAML literal 블록(|)으로 출력"""

    def _repr_str(dumper, data):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _LiteralDumper.add_representer(str, _repr_str)

    def _normalize(text: str) -> str:
        """프롬프트 재사용을 위해 들여쓰기 제거 및 트리밍"""
        return textwrap.dedent(text or "").strip()

    data = {
        "name": _normalize(profile.name),
        "persona": _normalize(profile.persona),
    }

    output = StringIO()
    yaml.dump(
        data,
        output,
        Dumper=_LiteralDumper,
        allow_unicode=True,
        sort_keys=False,
        width=120,
        default_flow_style=False,
    )
    return output.getvalue()


def save_profile_to_yaml(emp_no: str, profile: ProfessorProfile) -> Path:
    """생성된 프로파일을 emp_no.yaml로 저장 (기존 있으면 덮어씀)
    원자적 쓰기를 위해 임시 파일에 먼저 쓰고 원본 파일로 rename
    """
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    output_path = YAML_DIR / f"{emp_no}.yaml"
    temp_path = YAML_DIR / f"{emp_no}.yaml.tmp"

    yaml_text = profile_to_yaml_text(profile)
    
    # 임시 파일에 먼저 쓰기
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    
    # 원자적으로 원본 파일로 rename (임시 파일이 완전히 쓰여진 후)
    temp_path.replace(output_path)

    return output_path


async def save_profile_to_db(emp_no: str, yaml_text: str) -> None:
    """생성된 프로파일을 데이터베이스에 저장"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        file_path = f"/profiles/USR/{emp_no}.yaml"
        await conn.execute(
            "CALL sp_aita_profile_save($1, $2, $3, $4, $5)",
            emp_no,
            '*',
            'USR',
            yaml_text,
            file_path
        )
        print(f"DB 저장 완료: {emp_no}")
    finally:
        await conn.close()


async def run(emp_no: str):
    """전체 파이프라인 실행"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

    # 오라클 DB 연결
    conn = oracledb.connect(
        user=DATABASE2_CONFIG["user"],
        password=DATABASE2_CONFIG["password"],
        dsn=DATABASE2_CONFIG["dsn"],
    )

    try:
        prof_info = await fetch_professor_info(conn, emp_no)

        if not prof_info:
            raise ValueError(f"{emp_no} 에 대한 교수자 데이터가 없습니다.")

        profile = await generate_profile(prof_info)
        
        # YAML 텍스트 생성
        yaml_text = profile_to_yaml_text(profile)
        
        # 먼저 DB에 저장
        await save_profile_to_db(emp_no, yaml_text)
        
        # 그 다음 YAML 파일로 저장
        output_path = save_profile_to_yaml(emp_no, profile)
        print(f"YAML 파일 생성 완료: {output_path}")
    finally:
        conn.close()


def main(emp_no: str):
    """employee_number를 인자로 받아 실행"""
    asyncio.run(run(emp_no))


if __name__ == "__main__":
    # 인자를 주면 인자 값을 사용, 인자가 없으면 하드코딩된 기본값 사용
    if len(sys.argv) >= 2:
        emp_no = sys.argv[1]
    else:
        # EMP_NO = "45932"  # 이주성
        # EMP_NO = "43819"  # '재료과학개론II' 박종민
        emp_no = "35033"  # '무역결제론' 성시일

    main(emp_no)
