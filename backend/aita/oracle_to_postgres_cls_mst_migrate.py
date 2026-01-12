import os
import sys
import asyncio
import oracledb  # cx_Oracle 대신 oracledb 사용
import asyncpg
from sqlalchemy import create_engine, text
from config import DATABASE_CONFIG, DATABASE2_CONFIG


def fetch_oracle_data(open_year: str, open_term: str):
    """ Oracle에서 데이터를 조회하여 반환 """
    try:
        print("Oracle에서 데이터 조회 시작...")

        # SQLAlchemy를 이용해 Oracle 연결 (oracledb 사용)
        oracle_db_url = f"oracle+oracledb://{DATABASE2_CONFIG['user']}:{DATABASE2_CONFIG['password']}@210.115.225.121:2950/?service_name=ORA8.hallym.ac.kr"
        oracle_engine = create_engine(oracle_db_url)

        # SQL 실행
        query = text("""
                        SELECT LP.OPEN_YEAR||'-'||LP.OPEN_TERM||'-'||LP.COURSE_CODE||'-'||LP.COURSE_CODE_NO||'-'||LP.BUNBAN_CODE AS cls_id
                            ,LP.COURSE_CODE AS subj_cd
                            ,LC.COURSE_NAME AS cls_nm
                            ,LC.COURSE_ENAME  AS cls_nm_en
                            ,LP.BUNBAN_CODE AS cls_sec
                            ,LP.MEMBER_KEY  AS user_id
                            ,LP.OPEN_YEAR AS cls_yr
                            ,LP.OPEN_TERM AS cls_smt
                            ,LP.COURSE_CODE_NO AS cls_grd
                            --,LP.VISIBLE
                            ,SYSDATE AS ins_dt
                            ,SYSDATE AS upd_dt
                            ,'SMART' AS upload_type --2025.8.27 추가 
                        FROM HUIS.LMS_PROFESSOR LP
                        INNER JOIN HUIS.LMS_COURSE LC 
                                ON LP.COURSE_CODE = LC.COURSE_CODE 
                                AND LP.OPEN_YEAR = LC.OPEN_YEAR 
                                AND LP.OPEN_TERM = LC.OPEN_TERM 
                                AND LP.BUNBAN_CODE = LC.BUNBAN_CODE 
                        WHERE 1=1
                        AND LP.OPEN_YEAR = :open_year
                        AND LP.OPEN_TERM = :open_term
                        --AND LP.MEMBER_KEY = '20563'
                        AND LP.VISIBLE = 'Y' --수강처리구분 
                        AND LP.PROF_ORDER = '1' --교수인원순번(중복제거용)
        """)

        with oracle_engine.connect() as conn:
            result = conn.execute(query, {"open_year": open_year, "open_term": open_term})
            rows = []
            for row in result.fetchall():
                row = list(row)
                # cls_grd(8번째)와 문자열 컬럼을 str로 맞춤; 날짜 컬럼은 그대로 둠
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13]:
                    if idx < len(row):
                        row[idx] = str(row[idx]) if row[idx] is not None else None
                rows.append(tuple(row))

        print(f"Oracle 데이터 조회 완료! 총 {len(rows)}개")

        return rows
    except Exception as e:
        print(f"Oracle 연결 또는 쿼리 실행 중 오류 발생: {e}")
        return []

async def sync_postgres(data, open_year: str, open_term: str):
    """PostgreSQL 업서트 + 폐강(Oracle 미존재) SMART 데이터 삭제 (asyncpg)"""
    if not data:
        print("가져온 데이터가 없습니다.")
        return

    try:
        print("PostgreSQL 동기화 시작...")

        conn = await asyncpg.connect(**DATABASE_CONFIG)

        # Oracle에서 내려온 cls_id 목록
        oracle_cls_ids = [row[0] for row in data]

        # 폐강된 강좌 정리: Oracle에 없는 SMART 데이터는 삭제
        delete_query = """
            DELETE FROM cls_mst
             WHERE upload_type = 'SMART'
               AND cls_yr = $1
               AND cls_smt = $2
               AND cls_id NOT IN (SELECT unnest($3::text[]))
        """
        deleted = await conn.execute(delete_query, open_year, open_term, oracle_cls_ids)
        print(f"폐강/미존재 SMART 강좌 삭제 결과: {deleted}")

        # 기존 SMART 데이터의 rag_yn, rag_upd_dt, ins_dt 보존을 위해 미리 조회
        rag_rows = await conn.fetch(
            """
            SELECT cls_id, rag_yn, rag_upd_dt, ins_dt
              FROM cls_mst
             WHERE upload_type = 'SMART'
               AND cls_id = ANY($1::text[])
            """,
            oracle_cls_ids,
        )
        rag_map = {r["cls_id"]: (r["rag_yn"], r["rag_upd_dt"], r["ins_dt"]) for r in rag_rows}

        # 기존 SMART 데이터 중 이번 배치에 포함된 것만 삭제 후 재삽입 (ON CONFLICT 미사용)
        delete_existing_query = """
            DELETE FROM cls_mst
             WHERE upload_type = 'SMART'
               AND cls_id IN (SELECT unnest($1::text[]))
        """
        await conn.execute(delete_existing_query, oracle_cls_ids)

        # INSERT 시: 기존 rag_yn, rag_upd_dt, ins_dt 값이 있으면 유지, 없으면 기본값 사용
        insert_query = """
            INSERT INTO cls_mst (cls_id, course_id, cls_nm, cls_nm_en, cls_sec, user_id, cls_yr, cls_smt, cls_grd, rag_yn, rag_upd_dt, ins_dt, upd_dt, upload_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """

        data_with_rag = []
        for row in data:
            cls_id = row[0]
            # 기존 데이터가 있으면 rag_yn, rag_upd_dt, ins_dt 모두 유지 (rag_yn='N'이어도 ins_dt는 유지)
            # 기존 데이터가 없으면 기본값 사용 (rag_yn='N', rag_upd_dt=null, ins_dt=Oracle에서 가져온 값)
            existing = rag_map.get(cls_id)
            if existing:
                rag_yn, rag_upd_dt, ins_dt = existing
                # rag_yn이 'N'이어도 ins_dt는 기존 값 유지
            else:
                rag_yn, rag_upd_dt, ins_dt = "N", None, row[9]  # 새 데이터는 Oracle의 ins_dt 사용
            # 원본 row: [cls_id, subj_cd, cls_nm, cls_nm_en, cls_sec, user_id, cls_yr, cls_smt, cls_grd, ins_dt, upd_dt, upload_type]
            new_row = (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                rag_yn, rag_upd_dt, ins_dt, row[10], row[11]
            )
            data_with_rag.append(new_row)

        await conn.executemany(insert_query, data_with_rag)
        print(f"{len(data_with_rag)}개 데이터 삽입 완료!")

        await conn.close()
    except Exception as e:
        print(f"PostgreSQL 동기화 중 오류 발생: {e}")

if __name__ == "__main__":
    default_year = "2026"
    default_term = "10"

    # CLI 인자: python3 -m aita.oracle_to_postgres_cls_mst_migrate 2026 10
    args = sys.argv[1:]
    if len(args) >= 2:
        open_year, open_term = args[0], args[1]
    else:
        open_year, open_term = default_year, default_term

    print(f"데이터 적재 시작... (연도={open_year}, 학기={open_term})")

    data = fetch_oracle_data(open_year, open_term)
    asyncio.run(sync_postgres(data, open_year, open_term))
    print("완료!")
