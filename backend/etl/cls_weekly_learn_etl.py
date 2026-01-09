import asyncio
import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Sequence
from pathlib import Path

import asyncpg

# Add parent directory to sys.path to find config module
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_CONFIG, DATABASE3_CONFIG

# --- Schema ---

TABLE_NAME = "cls_weekly_learn"
MIRROR_TABLE_NAME = f"tutor_{TABLE_NAME}"

COLUMNS = [
    "cls_id",
    "week_num",
    "content_data",
    "ins_dt",
    "upd_dt",
]

async def ensure_tables_exist(conn: asyncpg.Connection, reset: bool = False):
    """Create tables if they don't exist."""
    if reset:
        print(f">>> Resetting tables (DROP & RECREATE) for {TABLE_NAME}...")
        await conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
        await conn.execute(f"DROP TABLE IF EXISTS {MIRROR_TABLE_NAME} CASCADE")

    # 1. Target Table
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            cls_id VARCHAR(50) NOT NULL,
            week_num INTEGER NOT NULL,
            content_data JSONB,
            ins_dt TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            upd_dt TIMESTAMP WITHOUT TIME ZONE,
            PRIMARY KEY (cls_id, week_num)
        );
    """)

    # 2. Mirror Table
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {MIRROR_TABLE_NAME} (
            cls_id VARCHAR(50) NOT NULL,
            week_num INTEGER NOT NULL,
            content_data JSONB,
            ins_dt TIMESTAMP WITHOUT TIME ZONE,
            upd_dt TIMESTAMP WITHOUT TIME ZONE,
            PRIMARY KEY (cls_id, week_num)
        );
    """)
    print(f"Ensured tables `{TABLE_NAME}` and `{MIRROR_TABLE_NAME}` exist.")

async def fetch_source_rows(pool: asyncpg.pool.Pool) -> List[asyncpg.Record]:
    """Fetch all rows from Source."""
    query = f"SELECT {', '.join(COLUMNS)} FROM {TABLE_NAME}"
    async with pool.acquire() as conn:
        return await conn.fetch(query)

def records_to_tuples(records: List[asyncpg.Record]) -> List[Tuple]:
    """Convert records to tuples, handling JSON serialization if needed."""
    return [tuple(record[col] for col in COLUMNS) for record in records]

async def refresh_mirror_table(conn: asyncpg.Connection, rows: Sequence[Sequence]) -> int:
    """Truncate and refill mirror table."""
    await conn.execute(f"TRUNCATE TABLE {MIRROR_TABLE_NAME}")
    if rows:
        # Note: asyncpg handles JSONB automatically if passed as string or dict? 
        # Usually dict/list for JSONB. records_to_tuples keeps them as python objects.
        # asyncpg should encode them to JSONB automatically.
        insert_sql = f"""
            INSERT INTO {MIRROR_TABLE_NAME} ({', '.join(COLUMNS)})
            VALUES ({', '.join(f'${i}' for i in range(1, len(COLUMNS) + 1))})
        """
        await conn.executemany(insert_sql, rows)
    return len(rows)

async def copy_to_target_table(conn: asyncpg.Connection) -> int:
    """Copy from Mirror to Target.
       Using PK (cls_id, week_num) to detect conflicts.
       We'll perform an UPSERT (INSERT ... ON CONFLICT DO UPDATE).
    """
    col_str = ', '.join(COLUMNS)
    # Exclude ins_dt from update to preserve original insertion time if desired?
    # Source has ins_dt, so we probably just want to overwrite with source data.
    
    update_set = ", ".join([f"{col} = EXCLUDED.{col}" for col in COLUMNS])

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} ({col_str})
        SELECT {col_str}
        FROM {MIRROR_TABLE_NAME}
        ON CONFLICT (cls_id, week_num) 
        DO UPDATE SET {update_set}
    """
    
    result = await conn.execute(insert_sql)
    # result format: "INSERT 0 <count>" or "INSERT 0 <count>" (upsert counts as insert in row count often, or 0 0 if nothing changed?)
    # Valid return from execute varies. 
    # Usually we just care it ran.
    return 0 

async def main() -> None:
    parser = argparse.ArgumentParser(description=f"ETL for {TABLE_NAME} migration.")
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
        print("Fetching from Source...")
        rows = await fetch_source_rows(source_pool)
        print(f"Fetched {len(rows)} rows.")

        print("Connecting to Target...")
        async with target_pool.acquire() as conn:
            # DDL outside transaction
            await ensure_tables_exist(conn, reset=args.reset_schema)

            async with conn.transaction():
                print("Refreshing Mirror Table...")
                count = await refresh_mirror_table(conn, records_to_tuples(rows))
                print(f"Mirror refreshed: {count} rows")

                print("Syncing to Target Table (Upsert)...")
                await copy_to_target_table(conn)
                print("Sync completed.")

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
        import traceback
        traceback.print_exc()
        raise
    finally:
        await source_pool.close()
        await target_pool.close()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Duration: {duration:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
