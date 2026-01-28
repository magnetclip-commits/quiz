import asyncio
import asyncpg
import requests
import json
import os
import sys
import argparse
import urllib3
from datetime import datetime
from pathlib import Path

# Add backend directory to sys.path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

# Suppress InsecureRequestWarning for self-signed certificates (verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils.milvus_util import construct_milvus_payload, send_milvus_ingest
from config import DATABASE_CONFIG

def ingest_to_milvus(payload: dict, file_id: str, dry_run: bool = False):
    """Sends a single file record to the Milvus Ingest API"""
    
    if dry_run:
        print(f"[{datetime.now()}] [DRY-RUN] Payload for {file_id}:")
        print(json.dumps(payload, indent=4, ensure_ascii=False))
        return True
        
    try:
        print(f"[{datetime.now()}] Sending {file_id} to Milvus...")
        success, response_text = send_milvus_ingest(payload)
        
        if success:
            print(f" -> Success: {response_text}")
            return True
        else:
            print(f" -> Failed: {response_text}")
            return False
    except Exception as e:
        print(f" -> Error calling Milvus API: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Daily Milvus Ingestion Sync Script")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without calling API or updating DB")
    parser.add_argument("--file_id", type=str, help="Process only a specific file_id")
    parser.add_argument("--all", action="store_true", help="Include files already synced to Milvus (milvus_yn='Y')")
    args = parser.parse_args()

    print(f"[{datetime.now()}] Starting Daily Milvus Sync Batch (Dry-run: {args.dry_run})")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        # Base query to find files to sync
        # Logic: Documents (M, N) -> FU03 (Upload/Download Comp)
        #        Videos (V) -> FS03 (STT Comp), EP03 (Embed Comp), SM03 (Summary Comp)
        query = """
            SELECT 
                f.file_id, f.cls_id, f.file_type_cd, f.file_nm, f.file_ext, 
                f.file_path, f.stt_file_path, f.file_size, f.week_num, c.user_id
            FROM cls_file_mst f
            JOIN cls_mst c ON f.cls_id = c.cls_id
            WHERE c.rag_yn IN ('N', 'Y')
            AND (
                (f.file_type_cd IN ('M', 'N') AND f.upload_status = 'FU03')
                OR (f.file_type_cd = 'V' AND f.upload_status IN ('FS03', 'EP03', 'SM03'))
            )
        """
        
        if args.file_id:
            query += " AND f.file_id = $1"
            rows = await conn.fetch(query, args.file_id)
        else:
            if not args.all:
                query += " AND (f.milvus_yn IS NULL OR f.milvus_yn != 'Y')"
            query += " ORDER BY f.dwld_comp_dt DESC"
            rows = await conn.fetch(query)
            
        print(f"Found {len(rows)} files to process.")
        
        if not rows:
            await conn.close()
            return

        for row in rows:
            file_id = row['file_id']
            file_type_cd = row['file_type_cd']
            stt_path = row['stt_file_path']
            
            # STRICT SAFETY CHECK: Never sync mp4 files. Must have stt_file_path for Videos.
            if file_type_cd == 'V' and (not stt_path or not stt_path.lower().endswith('.txt')):
                print(f"[{datetime.now()}] [Skipped] {file_id} is a video but has no valid STT transcript path.")
                continue

            payload = construct_milvus_payload(dict(row))
            success = ingest_to_milvus(payload, file_id, args.dry_run)
            
            if not args.dry_run:
                status = 'Y' if success else 'E'
                update_query = """
                    UPDATE cls_file_mst 
                    SET milvus_yn = $1, milvus_upd_dt = NOW() 
                    WHERE file_id = $2
                """
                await conn.execute(update_query, status, file_id)
                await asyncio.sleep(1) # Small delay
        
        await conn.close()
        
    except Exception as e:
        print(f"Fatal Error in Milvus Sync: {e}")

    print(f"[{datetime.now()}] Milvus Sync Batch Finished")

if __name__ == "__main__":
    asyncio.run(main())
