import asyncio
import os
import sys
import random
import string
import argparse
from datetime import datetime
from typing import List, Tuple, Sequence
from pathlib import Path

import asyncpg

# Add parent directory to sys.path to find config module
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_CONFIG, DATABASE3_CONFIG

# Columns to migrate
# Full Schema from Tutor
COLUMNS = [
    "file_id",
    "cls_id",
    "file_type_cd",
    "file_nm",
    "file_ext",
    "file_format",
    "upload_status",
    "file_size",
    "file_path",
    "stt_file_path",
    "dwld_start_dt",
    "dwld_comp_dt",
    "dwld_fail_dt",
    "stt_start_dt",
    "stt_comp_dt",
    "stt_fail_dt",
    "emb_start_dt",
    "emb_comp_dt",
    "emb_fail_dt",
    "file_del_req_dt",
    "emb_del_comp_dt",
    "week_num",
    "sumry_start_dt",
    "sumry_comp_dt",
    "sumry_fail_dt",
    "upload_type",
]

# Columns for mapping table
MAP_COLUMNS = ["tutor_file_id", "aita_file_id"]

async def ensure_tables_exist(conn: asyncpg.Connection, reset: bool = False):
    """Create helper tables if they don't exist.
       If reset is True, DROP tables first to ensure schema update.
    """
    if reset:
        print(">>> Resetting tables (DROP & RECREATE)...")
        await conn.execute("DROP TABLE IF EXISTS cls_file_mst CASCADE")
        await conn.execute("DROP TABLE IF EXISTS tutor_cls_file_mst CASCADE")
        # cls_file_id_map is usually safe to keep, but if we want a clean slate:
        # await conn.execute("DROP TABLE IF EXISTS cls_file_id_map CASCADE") 
        # For now, let's keep the map unless specifically asked, but user wants "same copy",
        # so keeping the map is fine, we just want the schema of the content tables to be correct.

    # 1. Target App Table: cls_file_mst
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS cls_file_mst (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20) NOT NULL,
            file_type_cd VARCHAR(10) NOT NULL,
            file_nm VARCHAR(255) NOT NULL,
            file_ext VARCHAR(10) NOT NULL,
            file_format VARCHAR(50) NOT NULL,
            upload_status VARCHAR(10) NOT NULL,
            file_size INTEGER NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            stt_file_path VARCHAR(500),
            dwld_start_dt TIMESTAMP WITHOUT TIME ZONE,
            dwld_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            dwld_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_start_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_start_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            file_del_req_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_del_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            week_num INTEGER,
            sumry_start_dt TIMESTAMP WITHOUT TIME ZONE,
            sumry_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            sumry_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            upload_type VARCHAR(20)
        );
    """)

    # 2. Mirror Table: tutor_cls_file_mst
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tutor_cls_file_mst (
            file_id VARCHAR(50) PRIMARY KEY,
            cls_id VARCHAR(20),
            file_type_cd VARCHAR(10),
            file_nm VARCHAR(255),
            file_ext VARCHAR(10),
            file_format VARCHAR(50),
            upload_status VARCHAR(10),
            file_size INTEGER,
            file_path VARCHAR(500),
            stt_file_path VARCHAR(500),
            dwld_start_dt TIMESTAMP WITHOUT TIME ZONE,
            dwld_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            dwld_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_start_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            stt_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_start_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            file_del_req_dt TIMESTAMP WITHOUT TIME ZONE,
            emb_del_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            week_num INTEGER,
            sumry_start_dt TIMESTAMP WITHOUT TIME ZONE,
            sumry_comp_dt TIMESTAMP WITHOUT TIME ZONE,
            sumry_fail_dt TIMESTAMP WITHOUT TIME ZONE,
            upload_type VARCHAR(20)
        );
    """)
    
    # 3. Mapping Table: cls_file_id_map
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS cls_file_id_map (
            tutor_file_id VARCHAR(50) PRIMARY KEY,
            aita_file_id VARCHAR(50) UNIQUE
        );
    """)
    print("Ensured full schema tables `cls_file_mst`, `tutor_cls_file_mst` exist.")


async def fetch_source_rows(pool: asyncpg.pool.Pool) -> List[asyncpg.Record]:
    """Fetch all rows from Source (Tutor) DB."""
    # Source Logic: upload_status != 'FD03' implies active/valid file
    query = f"SELECT {', '.join(COLUMNS)} FROM cls_file_mst WHERE upload_status != 'FD03'"
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return rows


async def refresh_tutor_cls_file_mst(
    conn: asyncpg.Connection, rows: Sequence[Sequence]
) -> int:
    """Truncate and refill tutor_cls_file_mst in Target DB."""
    insert_sql = f"""
        INSERT INTO tutor_cls_file_mst (
            {', '.join(COLUMNS)}
        ) VALUES (
            {', '.join(f'${i}' for i in range(1, len(COLUMNS) + 1))}
        )
    """
    # Truncate within the same transaction context provided by 'conn'
    await conn.execute("TRUNCATE TABLE tutor_cls_file_mst")
    if rows:
        await conn.executemany(insert_sql, rows)
    return len(rows)


def _build_aita_file_id(tutor_file_id: str, used_ids: set) -> str:
    """Generate a new ID for Assistant."""
    base = tutor_file_id
    while True:
        random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        candidate = f"{base}_{random_suffix}" 
        
        if len(candidate) > 50:
             candidate = f"{base[:45]}_{random_suffix}"
             
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate


async def insert_missing_file_id_map(conn: asyncpg.Connection) -> int:
    """Identify new tutor files and create mappings."""
    missing_sql = """
        SELECT t.file_id
        FROM tutor_cls_file_mst t
        LEFT JOIN cls_file_id_map m ON m.tutor_file_id = t.file_id
        WHERE m.tutor_file_id IS NULL
    """
    existing_sql = "SELECT file_id FROM cls_file_mst UNION SELECT aita_file_id FROM cls_file_id_map"
    
    missing_rows = await conn.fetch(missing_sql)
    if not missing_rows:
        return 0

    existing_aita_ids_rows = await conn.fetch(existing_sql)
    existing_ids = {row[0] for row in existing_aita_ids_rows}
    
    insert_values: List[Tuple[str, str]] = []
    for row in missing_rows:
        tutor_id = row["file_id"]
        aita_id = _build_aita_file_id(tutor_id, existing_ids)
        insert_values.append((tutor_id, aita_id))
        
    if insert_values:
        await conn.executemany(
            """
            INSERT INTO cls_file_id_map (tutor_file_id, aita_file_id)
            VALUES ($1, $2)
            """,
            insert_values,
        )
    return len(insert_values)


async def copy_to_cls_file_mst(conn: asyncpg.Connection) -> int:
    """Insert mapped files into actual cls_file_mst."""
    col_str = ', '.join(COLUMNS)
    select_cols = []
    for col in COLUMNS:
        if col == 'file_id':
            select_cols.append("m.aita_file_id AS file_id")
        else:
            select_cols.append(f"t.{col}")
            
    # Note: We are migrating only the columns present in Source.
    # If Target 'cls_file_mst' has NOT NULL constraints on other columns (e.g. ins_dt),
    # this might fail. We assume defaults or nulls are allowed.
    insert_sql = f"""
        INSERT INTO cls_file_mst (
            {col_str}
        )
        SELECT
            {', '.join(select_cols)}
        FROM tutor_cls_file_mst t
        JOIN cls_file_id_map m ON m.tutor_file_id = t.file_id
        LEFT JOIN cls_file_mst a ON a.file_id = m.aita_file_id
        WHERE a.file_id IS NULL
    """
    
    result = await conn.execute(insert_sql)
    return int(result.split()[-1])


def records_to_tuples(records: List[asyncpg.Record]) -> List[Tuple]:
    return [tuple(record[col] for col in COLUMNS) for record in records]


async def main() -> None:
    parser = argparse.ArgumentParser(description="ETL for cls_file_mst migration.")
    parser.add_argument("--dry-run", action="store_true", help="Run without committing changes.")
    parser.add_argument("--reset-schema", action="store_true", help="Drop and recreate content tables for schema update.")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    if args.dry_run:
        print(">>> DRY RUN MODE :: NO CHANGES WILL BE COMMITTED <<<")

    # Connect pools
    source_pool = await asyncpg.create_pool(**DATABASE3_CONFIG)
    
    # Force host to localhost (or DB_HOST env) to avoid container resolution on host
    # This matches the pattern in quiz_item_mst_etl.py
    target_config = {**DATABASE_CONFIG, "host": os.getenv("DB_HOST", "localhost")}
    target_pool = await asyncpg.create_pool(**target_config)

    try:
        # Fetch from Source (read-only)
        print("Fetching from Source...")
        source_records = await fetch_source_rows(source_pool)
        print(f"Fetched {len(source_records)} rows.")

        # Target operations in a single transaction
        print("Connecting to Target (Assistant)...")
        async with target_pool.acquire() as conn:
            # Execute table creation/reset outside of the main data transaction manually if needed, 
            # OR include it. 
            # Since we want reset to PERSIST, we do it here.
            await ensure_tables_exist(conn, reset=args.reset_schema)

            async with conn.transaction():
                # 2. Refresh Mirror
                print("Refreshing Mirror Table...")
                inserted_mirror = await refresh_tutor_cls_file_mst(
                    conn, records_to_tuples(source_records)
                )
                print(f"Mirror refreshed: {inserted_mirror} rows.")
                
                # 3. Map IDs
                print("Mapping IDs...")
                new_mappings = await insert_missing_file_id_map(conn)
                print(f"New mappings created: {new_mappings}")
                
                # 4. Copy to Target
                print("Copying to Target App Table...")
                inserted_target = await copy_to_cls_file_mst(conn)
                print(f"Inserted into `cls_file_mst`: {inserted_target} rows.")

                if args.dry_run:
                    print("\n>>> DRY RUN: Rolling back all changes...")
                    # Raising an exception will cause the transaction context manager to rollback
                    raise RuntimeError("Dry Run Rollback")

    except RuntimeError as e:
        if str(e) == "Dry Run Rollback":
            print(">>> Rollback successful. No changes made.")
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
        print(f"End Time: {end_time}")
        print(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
