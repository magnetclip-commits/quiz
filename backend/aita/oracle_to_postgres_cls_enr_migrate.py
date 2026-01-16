import asyncio
import os
import sys
import oracledb  # cx_Oracle 대신 oracledb 사용
import asyncpg
from sqlalchemy import create_engine, text
from config import DATABASE_CONFIG, DATABASE2_CONFIG

# Postgres 대상 테이블(필요 시 하나만 수정)
TARGET_TABLE = "cls_enr"
# 수강마스터 테이블(업로드 타입 확인용)
CLS_MST_TABLE = "cls_mst"


def fetch_oracle_data(open_year: str, open_term: str):
    """Oracle에서 수강 데이터 조회 (year/term 파라미터화)"""
    try:
        print(f"Oracle에서 데이터 조회 시작... year={open_year}, term={open_term}")

        oracle_db_url = (
            f"oracle+oracledb://{DATABASE2_CONFIG['user']}:{DATABASE2_CONFIG['password']}"
            "@210.115.225.121:2950/?service_name=ORA8.hallym.ac.kr"
        )
        oracle_engine = create_engine(oracle_db_url)

        query = text(
            """ -- 교수
                SELECT 'ENR_'||A.user_id||'_'||A.cls_id AS cls_enr_id
                    ,A.*
                FROM (
                        SELECT LP.MEMBER_KEY  AS user_id,
                            'O10' AS user_div, -- 교수
                            LP.OPEN_YEAR||'-'||LP.OPEN_TERM||'-'||LP.COURSE_CODE||'-'||LP.COURSE_CODE_NO||'-'||LP.BUNBAN_CODE AS cls_id,
                            NULL AS consent_yn,
                            NULL AS consent_dt,
                            SYSDATE AS ins_dt,
                            SYSDATE AS upd_dt
                        FROM HUIS.LMS_PROFESSOR LP
                        WHERE LP.OPEN_YEAR = :open_year
                        AND LP.OPEN_TERM = :open_term
                        AND LP.VISIBLE = 'Y' -- 수강처리구분
                        AND LP.PROF_ORDER = '1' -- 교수인원순번(중복제거용)
                        ) A
                UNION ALL
                -- 학생
                SELECT 'ENR_'||B.user_id||'_'||B.cls_id AS cls_enr_id
                    ,B.*
                FROM (
                    SELECT LS.MEMBER_KEY  AS user_id,
                        'S' AS user_div, -- 학생
                        LS.OPEN_YEAR||'-'||LS.OPEN_TERM||'-'||LS.COURSE_CODE||'-'||LS.COURSE_CODE_NO||'-'||LS.BUNBAN_CODE AS cls_id,
                        NULL AS consent_yn,
                        NULL AS consent_dt,
                        SYSDATE AS ins_dt,
                        SYSDATE AS upd_dt
                    FROM HUIS.LMS_STUDENT LS
                    WHERE LS.OPEN_YEAR = :open_year
                    AND LS.OPEN_TERM = :open_term
                    AND LS.VISIBLE = 'Y' -- 수강처리구분
                    )B
            """
        )

        with oracle_engine.connect() as conn:
            result = conn.execute(query, {"open_year": open_year, "open_term": open_term})
            rows = [row._mapping for row in result.fetchall()]

        print(f"Oracle 데이터 조회 완료! 총 {len(rows)}개")
        return rows
    except Exception as e:
        print(f"Oracle 연결 또는 쿼리 실행 중 오류 발생: {e}")
        return []


async def fetch_postgres_all_oracle_ids(conn: asyncpg.Connection, open_year: str, open_term: str) -> set[str]:
    """
    PostgreSQL에서 해당 year, term의 데이터 중
    cls_mst.upload_type = 'SMART' 인 수강건만 조회
    """
    rows = await conn.fetch(
        f"""
        SELECT e.cls_enr_id
          FROM {TARGET_TABLE} e
          JOIN {CLS_MST_TABLE} m
            ON e.cls_id = m.cls_id
         WHERE m.upload_type = 'SMART'
           AND e.cls_id LIKE $1 || '-' || $2 || '-%'
        """,
        open_year,
        open_term,
    )
    return {row["cls_enr_id"] for row in rows}


async def insert_new_records(conn: asyncpg.Connection, records):
    if not records:
        return

    insert_sql = f"""
        INSERT INTO {TARGET_TABLE}
            (cls_enr_id, user_id, user_div, cls_id, consent_yn, consent_dt, ins_dt, upd_dt)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8)
    """

    payload = [
        (
            r["cls_enr_id"],
            r["user_id"],
            r["user_div"],
            r["cls_id"],
            r["consent_yn"],
            r["consent_dt"],
            r["ins_dt"],
            r["upd_dt"],
        )
        for r in records
    ]

    await conn.executemany(insert_sql, payload)
    print(f"{len(records)}개 데이터 삽입 완료")


async def delete_missing_records(conn: asyncpg.Connection, cls_enr_ids, open_year: str, open_term: str):
    """Oracle에 없는 데이터 중 cls_mst.upload_type='SMART' 인 경우만 삭제"""
    if not cls_enr_ids:
        return

    delete_sql = f"""
        DELETE FROM {TARGET_TABLE} e
         WHERE e.cls_enr_id = ANY($1::text[])
           AND e.cls_id LIKE $2 || '-' || $3 || '-%'
           AND EXISTS (
                SELECT 1
                  FROM {CLS_MST_TABLE} m
                 WHERE m.cls_id = e.cls_id
                   AND m.upload_type = 'SMART'
           )
    """
    await conn.execute(delete_sql, list(cls_enr_ids), open_year, open_term)
    print(f"{len(cls_enr_ids)}개 데이터 삭제 완료")


async def sync_enrollments(open_year: str, open_term: str):
    oracle_rows = fetch_oracle_data(open_year, open_term)
    oracle_ids = {row["cls_enr_id"] for row in oracle_rows} if oracle_rows else set()

    pg_conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        # PostgreSQL에서 해당 year, term의 모든 데이터 조회
        all_pg_ids = await fetch_postgres_all_oracle_ids(pg_conn, open_year, open_term)
        
        # 삽입: Oracle에는 있지만 PostgreSQL에는 없는 것
        to_insert = [row for row in oracle_rows if row["cls_enr_id"] not in all_pg_ids]
        
        # 삭제: PostgreSQL에는 있지만 Oracle 목록에 없는 것
        # (단, cls_mst.upload_type = 'SMART' 조건으로 필터링되므로
        #  SMART가 아닌 데이터는 조회/삭제 대상에서 제외됨)
        to_delete = all_pg_ids - oracle_ids

        await insert_new_records(pg_conn, to_insert)
        await delete_missing_records(pg_conn, to_delete, open_year, open_term)
    finally:
        await pg_conn.close()


if __name__ == "__main__":
    # 우선 순위: CLI 인자 > 환경변수 > 하드코딩 기본값
    default_year = "2026"
    default_term = "10"

    if len(sys.argv) >= 3:
        target_year = sys.argv[1]
        target_term = sys.argv[2]
    else:
        target_year = os.getenv("OPEN_YEAR", default_year)
        target_term = os.getenv("OPEN_TERM", default_term)

    print(f"수강 데이터 동기화 시작... year={target_year}, term={target_term}")
    asyncio.run(sync_enrollments(target_year, target_term))
    print("완료!")