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
from dotenv import load_dotenv

# Add backend directory to sys.path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

# Load .env file
load_dotenv(dotenv_path=backend_dir / ".env")

# Suppress InsecureRequestWarning for self-signed certificates (verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import DATABASE_CONFIG

# Configuration from Environment Variables
MILVUS_INGEST_URL = os.getenv("MILVUS_INGEST_URL", "https://kaai-jobapi:5680/jobs")
TENANT_ID = os.getenv("MILVUS_TENANT_ID", "7c934d46-9c90-4327-bcc5-60d464131b06")
CALLBACK_URL_BASE = os.getenv("MILVUS_CALLBACK_URL", "http://172.17.0.1:56789/callback")
X_API_KEY = os.getenv("MILVUS_API_KEY", "541e23d279a27ecf50902a78c43bd9861175c0b0386a5ecb48c6b3cf0a092608")

def construct_payload(file_record: dict):
    """Constructs the JSON payload for the Milvus Ingest API"""
    file_id = file_record['file_id']
    cls_id = file_record['cls_id']
    user_id = file_record['user_id']
    file_nm = file_record['file_nm']
    file_ext = file_record['file_ext']
    file_path = file_record['file_path']
    file_size = file_record['file_size']
    week_num = file_record['week_num'] or 1
    file_type_cd = file_record['file_type_cd']
    
    # Final user_id (Student/Faculty ID as requested)
    final_user_id = user_id
    
    # Mapping content type: M (Material) -> lecture_material, Others -> bulletin_board
    content_type = "lecture_material" if file_type_cd == 'M' else "bulletin_board"
    
    # Fixed S3 prefix as requested by admin
    S3_FIXED_PREFIX = "s3://c11ebc288e44a7952a69876b2c834ff44ac7b00f"
    full_file_name = f"{file_nm}.{file_ext}" if not file_nm.endswith(file_ext) else file_nm
    storage_url = f"{S3_FIXED_PREFIX}/{full_file_name}"
    
    # Payload structured according to the JobManager requirement
    return {
        "command": "ingest",
        "tenant_id": TENANT_ID,
        "input": {
            "user_id": final_user_id,
            "cls_id": cls_id,
            "file_id": file_id,
            "storage_url": storage_url,
            "metadata": {
                "files": [
                    {
                        "filename": full_file_name,
                        "saved_pathname": file_path,
                        "size": file_size
                    }
                ],
                "week": week_num,
                "title": full_file_name,
                "content_type": content_type,
                "embedding_date": datetime.now().strftime("%Y-%m-%d")
            },
            "config": {
                "chunk_size": 250,
                "chunk_overlap": 50
            }
        },
        "callback": {
            "url": CALLBACK_URL_BASE,
            "id": file_id
        }
    }

def ingest_to_milvus(payload: dict, file_id: str, dry_run: bool = False):
    """Sends a single file record to the Milvus Ingest API (Synchronous using requests)"""
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": X_API_KEY
    }
    
    if dry_run:
        print(f"[{datetime.now()}] [DRY-RUN] Payload for {file_id}:")
        print(json.dumps(payload, indent=4, ensure_ascii=False))
        return True
        
    try:
        print(f"[{datetime.now()}] Sending {file_id} to Milvus ({MILVUS_INGEST_URL})...")
        # verify=False is used because JobManager uses a self-signed certificate
        resp = requests.post(MILVUS_INGEST_URL, headers=headers, json=payload, timeout=30.0, verify=False)
        
        if resp.status_code in [200, 201]:
            print(f" -> Success: {resp.text}")
            return True
        else:
            print(f" -> Failed: Status {resp.status_code}, {resp.text}")
            return False
    except Exception as e:
        print(f" -> Error calling Milvus API: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Daily Milvus Ingestion Sync Script")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without calling API or updating DB")
    parser.add_argument("--file_id", type=str, help="Process only a specific file_id")
    args = parser.parse_args()

    print(f"[{datetime.now()}] Starting Daily Milvus Sync Batch (Dry-run: {args.dry_run})")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        # Base query to find files to sync
        query = """
            SELECT 
                f.file_id, f.cls_id, f.file_type_cd, f.file_nm, f.file_ext, 
                f.file_path, f.file_size, f.week_num, c.user_id
            FROM cls_file_mst f
            JOIN cls_mst c ON f.cls_id = c.cls_id
            WHERE c.rag_yn IN ('N', 'Y')
            AND f.upload_status = 'FU03'
        """
        
        if args.file_id:
            query += " AND f.file_id = $1"
            rows = await conn.fetch(query, args.file_id)
        else:
            query += " AND (f.milvus_yn IS NULL OR f.milvus_yn != 'Y') ORDER BY f.dwld_comp_dt DESC"
            rows = await conn.fetch(query)
            
        print(f"Found {len(rows)} files to process.")
        
        if not rows:
            await conn.close()
            return

        for row in rows:
            file_id = row['file_id']
            payload = construct_payload(dict(row))
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
