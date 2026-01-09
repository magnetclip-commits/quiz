import asyncio
import os
from datetime import datetime
import random
import string
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import asyncpg

# 상위 디렉토리를 sys.path에 추가하여 config 모듈을 찾을 수 있도록 함
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_CONFIG, DATABASE3_CONFIG

# 공통으로 사용하는 컬럼 목록을 한 곳에서 정의해 재사용
COLUMNS = [
    "item_id",
    "course_id",
    "item_content",
    "item_choices",
    "item_answer",
    "item_explain",
    "item_type_cd",
    "item_diff_cd",
    "file_path",
    "ins_user_id",
    "ins_dt",
    "upd_user_id",
    "upd_dt",
    "del_yn",
    "grading_note",
]


async def fetch_source_rows(pool: asyncpg.pool.Pool) -> List[asyncpg.Record]:
    # DATABASE3.quiz_item_mst 원본 조회
    query = f"SELECT {', '.join(COLUMNS)} FROM quiz_item_mst"
    async with pool.acquire() as conn:
        return await conn.fetch(query)


async def refresh_tutor_quiz_item_mst(
    pool: asyncpg.pool.Pool, rows: Sequence[Sequence]
) -> int:
    # tutor_quiz_item_mst를 비우고 원본 전체 적재
    insert_sql = f"""
        INSERT INTO tutor_quiz_item_mst (
            {', '.join(COLUMNS)}
        ) VALUES (
            {', '.join(f'${i}' for i in range(1, len(COLUMNS) + 1))}
        )
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("TRUNCATE TABLE tutor_quiz_item_mst")
            if rows:
                await conn.executemany(insert_sql, rows)
    return len(rows)


def _build_aita_item_id(tutor_item_id: str, used_ids: set) -> str:
    # 하이픈을 언더바로 바꾸고 랜덤 6자리로 suffix 생성
    base = tutor_item_id.replace("-", "_")
    prefix = base[:-6] if len(base) > 6 else ""

    while True:
        random_suffix = "".join(random.choices(string.ascii_uppercase, k=6))
        candidate = f"{prefix}{random_suffix}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate


async def insert_missing_quiz_item_id_map(pool: asyncpg.pool.Pool) -> int:
    # tutor에만 있는 item을 quiz_item_id_map에 신규 매핑 삽입
    missing_sql = """
        SELECT t.item_id
        FROM tutor_quiz_item_mst t
        LEFT JOIN quiz_item_id_map m ON m.tutor_item_id = t.item_id
        WHERE m.tutor_item_id IS NULL
    """
    existing_sql = "SELECT aita_item_id FROM quiz_item_id_map"
    async with pool.acquire() as conn:
        missing_rows = await conn.fetch(missing_sql)
        existing_aita_ids = {row["aita_item_id"] for row in await conn.fetch(existing_sql)}

        if not missing_rows:
            return 0

        insert_values: List[Tuple[str, str]] = []
        for row in missing_rows:
            tutor_item_id = row["item_id"]
            aita_item_id = _build_aita_item_id(tutor_item_id, existing_aita_ids)
            insert_values.append((tutor_item_id, aita_item_id))

        await conn.executemany(
            """
            INSERT INTO quiz_item_id_map (tutor_item_id, aita_item_id)
            VALUES ($1, $2)
            """,
            insert_values,
        )
        return len(insert_values)


async def copy_to_aita_quiz_item_mst(pool: asyncpg.pool.Pool) -> int:
    # 매핑된 aita_item_id 기준 aita_quiz_item_mst에 미존재 건만 적재
    insert_sql = f"""
        INSERT INTO aita_quiz_item_mst (
            {', '.join(COLUMNS)}
        )
        SELECT
            m.aita_item_id AS item_id,
            t.course_id,
            t.item_content,
            t.item_choices,
            t.item_answer,
            t.item_explain,
            t.item_type_cd,
            t.item_diff_cd,
            t.file_path,
            t.ins_user_id,
            t.ins_dt,
            t.upd_user_id,
            t.upd_dt,
            t.del_yn
        FROM tutor_quiz_item_mst t
        JOIN quiz_item_id_map m ON m.tutor_item_id = t.item_id
        LEFT JOIN aita_quiz_item_mst a ON a.item_id = m.aita_item_id
        WHERE a.item_id IS NULL
    """
    async with pool.acquire() as conn:
        result = await conn.execute(insert_sql)
    # asyncpg returns "INSERT <count>"
    return int(result.split()[-1])


def records_to_tuples(records: Iterable[asyncpg.Record]) -> List[Tuple]:
    return [tuple(record[column] for column in COLUMNS) for record in records]


async def main() -> None:
    start_time = datetime.now()
    print(f"시작시간: {start_time.isoformat(sep=' ', timespec='seconds')}")
    source_pool = await asyncpg.create_pool(**DATABASE3_CONFIG)
    # DATABASE_CONFIG를 복사하고 HOST만 DB_HOST 환경변수에서 읽어옴 (기본값: localhost)
    target_config = {**DATABASE_CONFIG, "host": os.getenv("DB_HOST", "localhost")}
    target_pool = await asyncpg.create_pool(**target_config)

    try:
        source_records = await fetch_source_rows(source_pool)
        inserted_tutor = await refresh_tutor_quiz_item_mst(
            target_pool, records_to_tuples(source_records)
        )
        inserted_map = await insert_missing_quiz_item_id_map(target_pool)
        inserted_aita = await copy_to_aita_quiz_item_mst(target_pool)

        print(f"tutor_quiz_item_mst 새로 적재: {inserted_tutor}건")
        print(f"quiz_item_id_map 신규 매핑: {inserted_map}건")
        print(f"aita_quiz_item_mst 신규 적재: {inserted_aita}건")
    finally:
        await source_pool.close()
        await target_pool.close()
        end_time = datetime.now()
        print(f"종료시간: {end_time.isoformat(sep=' ', timespec='seconds')}")
        duration = (end_time - start_time).total_seconds()
        print(f"소요시간: {duration:.1f}초")


if __name__ == "__main__":
    asyncio.run(main())