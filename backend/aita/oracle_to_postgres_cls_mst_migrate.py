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

        # 기존 SMART 데이터의 rag_yn, rag_upd_dt, ins_dt, upd_dt 및 비교 대상 컬럼 조회
        existing_rows = await conn.fetch(
            """
            SELECT cls_id, course_id, cls_nm, cls_nm_en, cls_sec, user_id, cls_yr, cls_smt, cls_grd,
                   rag_yn, rag_upd_dt, ins_dt, upd_dt
              FROM cls_mst
             WHERE upload_type = 'SMART'
               AND cls_id = ANY($1::text[])
            """,
            oracle_cls_ids,
        )
        
        # 기존 데이터를 cls_id 기준으로 매핑
        existing_map = {}
        for r in existing_rows:
            cls_id = r["cls_id"]
            existing_map[cls_id] = {
                "course_id": r["course_id"],
                "cls_nm": r["cls_nm"],
                "cls_nm_en": r["cls_nm_en"],
                "cls_sec": r["cls_sec"],
                "user_id": r["user_id"],
                "cls_yr": r["cls_yr"],
                "cls_smt": r["cls_smt"],
                "cls_grd": r["cls_grd"],
                "rag_yn": r["rag_yn"],
                "rag_upd_dt": r["rag_upd_dt"],
                "ins_dt": r["ins_dt"],
                "upd_dt": r["upd_dt"],
            }

        # 변경 감지 및 삽입할 데이터만 수집
        insert_query = """
            INSERT INTO cls_mst (cls_id, course_id, cls_nm, cls_nm_en, cls_sec, user_id, cls_yr, cls_smt, cls_grd, rag_yn, rag_upd_dt, ins_dt, upd_dt, upload_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """

        data_to_insert = []  # 삽입할 데이터만 수집
        cls_ids_to_delete = []  # 삭제할 cls_id 목록 (변경된 데이터만)
        updated_count = 0
        unchanged_count = 0
        new_count = 0
        
        for row in data:
            cls_id = row[0]
            existing = existing_map.get(cls_id)
            
            if existing:
                # 기존 데이터가 있는 경우 - 변경 여부 확인
                rag_yn, rag_upd_dt, ins_dt = existing["rag_yn"], existing["rag_upd_dt"], existing["ins_dt"]
                
                # 데이터 변경 여부 확인 (비교 대상: course_id, cls_nm, cls_nm_en, cls_sec, user_id, cls_yr, cls_smt, cls_grd)
                is_changed = (
                    str(existing["course_id"]) != str(row[1]) or
                    str(existing["cls_nm"] or "") != str(row[2] or "") or
                    str(existing["cls_nm_en"] or "") != str(row[3] or "") or
                    str(existing["cls_sec"] or "") != str(row[4] or "") or
                    str(existing["user_id"] or "") != str(row[5] or "") or
                    str(existing["cls_yr"] or "") != str(row[6] or "") or
                    str(existing["cls_smt"] or "") != str(row[7] or "") or
                    str(existing["cls_grd"] or "") != str(row[8] or "")
                )
                
                if is_changed:
                    # 변경된 데이터만 삭제/재삽입 대상에 추가
                    upd_dt = row[10]  # 변경되었으면 새 upd_dt 사용
                    updated_count += 1
                    cls_ids_to_delete.append(cls_id)
                    new_row = (
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                        rag_yn, rag_upd_dt, ins_dt, upd_dt, row[11]
                    )
                    data_to_insert.append(new_row)
                else:
                    # 변경되지 않았으면 삭제/재삽입 건너뛰기
                    unchanged_count += 1
            else:
                # 새 데이터는 항상 삽입
                rag_yn, rag_upd_dt, ins_dt = "N", None, row[9]  # 새 데이터는 Oracle의 ins_dt 사용
                upd_dt = row[10]  # 새 데이터는 새 upd_dt 사용
                new_count += 1
                new_row = (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                    rag_yn, rag_upd_dt, ins_dt, upd_dt, row[11]
                )
                data_to_insert.append(new_row)
        
        print(f"데이터 변경 감지: {updated_count}개 변경됨, {unchanged_count}개 변경 없음, {new_count}개 새 데이터")

        # 변경된 데이터만 삭제
        if cls_ids_to_delete:
            delete_existing_query = """
                DELETE FROM cls_mst
                 WHERE upload_type = 'SMART'
                   AND cls_id = ANY($1::text[])
            """
            await conn.execute(delete_existing_query, cls_ids_to_delete)
            print(f"{len(cls_ids_to_delete)}개 변경된 데이터 삭제 완료")

        # 변경된 데이터와 새 데이터만 삽입
        if data_to_insert:
            await conn.executemany(insert_query, data_to_insert)
            print(f"{len(data_to_insert)}개 데이터 삽입 완료!")
        else:
            print("삽입할 데이터가 없습니다.")

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
