import asyncio
import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Sequence
from pathlib import Path

import asyncpg

# Add parent directory to sys.path to find config module
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_CONFIG, DATABASE3_CONFIG

# --- Schemas ---

LMS_FILE_WEEKLY_COLS = [
    "file_id", "cls_id", "user_id", "course_id", "week_num",
    "material_id", "material_nm", "material_url", "material_type", "ins_dt"
]

LMS_FILE_NOTICE_COLS = [
    "file_id", "cls_id", "user_id", "course_id", "page_num",
    "notice_num", "notice_nm", "notice_url", "ins_dt"
]

COMM_SMT_WEEK_COLS = [
    "yr", "smt", "yr_smt", "week_num", "week_nm",
    "week_full_nm", "week_start_date", "week_end_date", "week_descp", "ins_dt"
]

async def ensure_tables_exist(conn: asyncpg.Connection, reset: bool = False):
    """Create tables if they don't exist.
       Includes Target tables and Mirror tables for file-related tables.
    """
    if reset:
        print(">>> Resetting tables (DROP & RECREATE)...")
        await conn.execute("DROP TABLE IF EXISTS lms_file_weekly CASCADE")
        await conn.execute("DROP TABLE IF EXISTS tutor_lms_file_weekly CASCADE")
        await conn.execute("DROP TABLE IF EXISTS lms_file_notice CASCADE")
        await conn.execute("DROP TABLE IF EXISTS tutor_lms_file_notice CASCADE")
        await conn.execute("DROP TABLE IF EXISTS comm_smt_week CASCADE")

    # 1. lms_file_weekly
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS lms_file_weekly (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20) NOT NULL,
            user_id VARCHAR(50) NOT NULL,
            course_id VARCHAR(50) NOT NULL,
            week_num INTEGER NOT NULL,
            material_id VARCHAR(20) NOT NULL,
            material_nm VARCHAR(255) NOT NULL,
            material_url TEXT NOT NULL,
            material_type VARCHAR(50),
            ins_dt TIMESTAMP WITHOUT TIME ZONE
        );
    """)
    # Mirror for lms_file_weekly
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tutor_lms_file_weekly (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20),
            user_id VARCHAR(50),
            course_id VARCHAR(50),
            week_num INTEGER,
            material_id VARCHAR(20),
            material_nm VARCHAR(255),
            material_url TEXT,
            material_type VARCHAR(50),
            ins_dt TIMESTAMP WITHOUT TIME ZONE
        );
    """)

    # 2. lms_file_notice
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS lms_file_notice (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20) NOT NULL,
            user_id VARCHAR(50) NOT NULL,
            course_id VARCHAR(50) NOT NULL,
            page_num INTEGER NOT NULL,
            notice_num INTEGER NOT NULL,
            notice_nm VARCHAR(255) NOT NULL,
            notice_url TEXT NOT NULL,
            ins_dt TIMESTAMP WITHOUT TIME ZONE
        );
    """)
    # Mirror for lms_file_notice
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tutor_lms_file_notice (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20),
            user_id VARCHAR(50),
            course_id VARCHAR(50),
            page_num INTEGER,
            notice_num INTEGER,
            notice_nm VARCHAR(255),
            notice_url TEXT,
            ins_dt TIMESTAMP WITHOUT TIME ZONE
        );
    """)

    # 3. comm_smt_week (Simple copy, no mapping needed usually unless week_id exists)
    # Based on schema provided, no ID to map? No file_id.
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS comm_smt_week (
             yr VARCHAR(4) NOT NULL,
             smt VARCHAR(2) NOT NULL,
             yr_smt VARCHAR(10) NOT NULL,
             week_num INTEGER NOT NULL,
             week_nm VARCHAR(10) NOT NULL,
             week_full_nm VARCHAR(50) NOT NULL,
             week_start_date DATE NOT NULL,
             week_end_date DATE NOT NULL,
             week_descp TEXT,
             ins_dt TIMESTAMP WITHOUT TIME ZONE
        );
    """)
    print("Ensured tables exist.")

# --- Generic Fetch ---

async def fetch_all(pool: asyncpg.pool.Pool, table: str, columns: List[str]) -> List[asyncpg.Record]:
    query = f"SELECT {', '.join(columns)} FROM {table}"
    async with pool.acquire() as conn:
        return await conn.fetch(query)

def records_to_tuples(records: List[asyncpg.Record], columns: List[str]) -> List[Tuple]:
    return [tuple(record[col] for col in columns) for record in records]

# --- Migration Logics ---

async def migrate_comm_smt_week(conn: asyncpg.Connection, rows: Sequence[Sequence]):
    """Direct truncate and insert for comm_smt_week."""
    print(f"Migrating comm_smt_week ({len(rows)} rows)...")
    await conn.execute("TRUNCATE TABLE comm_smt_week")
    
    if not rows:
        return
        
    insert_sql = f"""
        INSERT INTO comm_smt_week ({', '.join(COMM_SMT_WEEK_COLS)})
        VALUES ({', '.join(f'${i}' for i in range(1, len(COMM_SMT_WEEK_COLS) + 1))})
    """
    await conn.executemany(insert_sql, rows)

async def migrate_lms_file_weekly(conn: asyncpg.Connection, rows: Sequence[Sequence]):
    """Mirror then Map Insert for lms_file_weekly."""
    print(f"Migrating lms_file_weekly ({len(rows)} rows)...")
    
    # 1. Fill Mirror
    await conn.execute("TRUNCATE TABLE tutor_lms_file_weekly")
    if rows:
        insert_mirror_sql = f"""
            INSERT INTO tutor_lms_file_weekly ({', '.join(LMS_FILE_WEEKLY_COLS)})
            VALUES ({', '.join(f'${i}' for i in range(1, len(LMS_FILE_WEEKLY_COLS) + 1))})
        """
        await conn.executemany(insert_mirror_sql, rows)
    
    # 2. Insert into Target with Mapping
    # We join with cls_file_id_map to translate file_id
    cols_without_file_id = [c for c in LMS_FILE_WEEKLY_COLS if c != 'file_id']
    select_cols = [f"t.{c}" for c in cols_without_file_id]
    
    # Target Insert
    # We ignore conflicts or duplicates? Schema says file_id is PK.
    # We use ON CONFLICT DO NOTHING for safety if re-running without total reset.
    insert_target_sql = f"""
        INSERT INTO lms_file_weekly (file_id, {', '.join(cols_without_file_id)})
        SELECT 
            m.aita_file_id, 
            {', '.join(select_cols)}
        FROM tutor_lms_file_weekly t
        JOIN cls_file_id_map m ON m.tutor_file_id = t.file_id
        ON CONFLICT (file_id) DO NOTHING
    """
    result = await conn.execute(insert_target_sql)
    print(f"  -> Inserted/Ignored: {result}")

async def migrate_lms_file_notice(conn: asyncpg.Connection, rows: Sequence[Sequence]):
    """Mirror then Map Insert for lms_file_notice."""
    print(f"Migrating lms_file_notice ({len(rows)} rows)...")
    
    # 1. Fill Mirror
    await conn.execute("TRUNCATE TABLE tutor_lms_file_notice")
    if rows:
        insert_mirror_sql = f"""
            INSERT INTO tutor_lms_file_notice ({', '.join(LMS_FILE_NOTICE_COLS)})
            VALUES ({', '.join(f'${i}' for i in range(1, len(LMS_FILE_NOTICE_COLS) + 1))})
        """
        await conn.executemany(insert_mirror_sql, rows)
    
    # 2. Insert into Target with Mapping
    cols_without_file_id = [c for c in LMS_FILE_NOTICE_COLS if c != 'file_id']
    select_cols = [f"t.{c}" for c in cols_without_file_id]
    
    insert_target_sql = f"""
        INSERT INTO lms_file_notice (file_id, {', '.join(cols_without_file_id)})
        SELECT 
            m.aita_file_id, 
            {', '.join(select_cols)}
        FROM tutor_lms_file_notice t
        JOIN cls_file_id_map m ON m.tutor_file_id = t.file_id
        ON CONFLICT (file_id) DO NOTHING
    """
    result = await conn.execute(insert_target_sql)
    print(f"  -> Inserted/Ignored: {result}")

async def main() -> None:
    parser = argparse.ArgumentParser(description="ETL for LMS tables migration.")
    parser.add_argument("--dry-run", action="store_true", help="Run without committing changes.")
    parser.add_argument("--reset-schema", action="store_true", help="Drop and recreate tables.")
    args = parser.parse_args()

    start_time = datetime.now()
    if args.dry_run:
        print(">>> DRY RUN MODE :: NO CHANGES WILL BE COMMITTED <<<")
    
    source_pool = await asyncpg.create_pool(**DATABASE3_CONFIG)
    
    target_config = {**DATABASE_CONFIG, "host": os.getenv("DB_HOST", "localhost")}
    target_pool = await asyncpg.create_pool(**target_config)

    try:
        # Fetch Source Data
        print("Fetching from Source...")
        weekly_rows = await fetch_all(source_pool, "lms_file_weekly", LMS_FILE_WEEKLY_COLS)
        notice_rows = await fetch_all(source_pool, "lms_file_notice", LMS_FILE_NOTICE_COLS)
        comm_rows = await fetch_all(source_pool, "comm_smt_week", COMM_SMT_WEEK_COLS)
        print(f"Fetched: Weekly={len(weekly_rows)}, Notice={len(notice_rows)}, Comm={len(comm_rows)}")

        print("Connecting to Target...")
        async with target_pool.acquire() as conn:
            # DDL outside transaction to persist schema
            await ensure_tables_exist(conn, reset=args.reset_schema)

            async with conn.transaction():
                # Migration
                await migrate_comm_smt_week(conn, records_to_tuples(comm_rows, COMM_SMT_WEEK_COLS))
                await migrate_lms_file_weekly(conn, records_to_tuples(weekly_rows, LMS_FILE_WEEKLY_COLS))
                await migrate_lms_file_notice(conn, records_to_tuples(notice_rows, LMS_FILE_NOTICE_COLS))

                if args.dry_run:
                    print("\n>>> DRY RUN: Rolling back data changes...")
                    raise RuntimeError("Dry Run Rollback")

    except RuntimeError as e:
        if str(e) == "Dry Run Rollback":
            print(">>> Rollback successful. No data changes made (Tables persisted).")
        else:
            raise
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        await source_pool.close()
        await target_pool.close()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Duration: {duration:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
