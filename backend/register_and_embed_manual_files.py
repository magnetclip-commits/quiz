import asyncio
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
import asyncpg

# Add parent directory to sys.path to find config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_CONFIG
from embedding_m import process_and_embed_files

# Configuration
MANUAL_FILE_ROOT = "/app/tutor/download_files/manual_test"

def generate_file_id(cls_id: str, file_nm: str, file_reg_dt: datetime) -> str:
    """Matches the logic in routers/file.py"""
    hash_string = f"{cls_id}{file_nm}{file_reg_dt.isoformat()}"
    hash_object = hashlib.sha256(hash_string.encode())
    hash_value = hash_object.hexdigest()[:6]
    return f"{cls_id}-{hash_value}"

async def register_and_embed():
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        if not os.path.exists(MANUAL_FILE_ROOT):
            print(f"Error: Directory {MANUAL_FILE_ROOT} does not exist.")
            return

        cls_to_process = {}

        # 1. Scan cls_id directories
        for cls_id in os.listdir(MANUAL_FILE_ROOT):
            cls_base_path = os.path.join(MANUAL_FILE_ROOT, cls_id)
            if not os.path.isdir(cls_base_path):
                continue

            print(f"\n--- Processing Class: {cls_id} ---")
            
            # Verify cls_id exists in cls_mst
            cls_info = await conn.fetchrow("SELECT user_id FROM cls_mst WHERE cls_id = $1", cls_id)
            if not cls_info:
                print(f"Warning: Class {cls_id} not found in cls_mst. Skipping.")
                continue
            
            user_id = cls_info['user_id']

            # 2. Walk through the directory (handles week_1, week_2, etc.)
            for root, dirs, files in os.walk(cls_base_path):
                for file_name in files:
                    if file_name.startswith('.'): continue
                    
                    full_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(full_path, cls_base_path)
                    
                    # Extract week_num from folder name (e.g., E-2025-2-001/week_1/...)
                    # We assume structure: {cls_id}/week_{n}/filename or {cls_id}/filename
                    week_num = 1
                    parts = relative_path.split(os.sep)
                    for part in parts:
                        if part.startswith('week_'):
                            try:
                                week_num = int(part.replace('week_', ''))
                                break
                            except ValueError:
                                pass

                    file_nm, file_ext = os.path.splitext(file_name)
                    file_ext = file_ext.lstrip('.').lower()
                    
                    # Skip video files if we only want materials, or handle them accordingly
                    if file_ext in ['mp4', 'm4v', 'avi', 'mov']:
                        print(f"Skipping video file: {file_name}")
                        continue

                    # Check if already registered
                    row = await conn.fetchrow(
                        "SELECT file_id, upload_status FROM cls_file_mst WHERE file_path = $1", full_path
                    )
                    
                    if row:
                        file_id = row['file_id']
                        status = row['upload_status']
                        print(f"  [Existing] {file_name} (Status: {status})")
                    else:
                        file_size = os.path.getsize(full_path)
                        reg_dt = datetime.now()
                        file_id = generate_file_id(cls_id, file_nm, reg_dt)
                        
                        print(f"  [New] Registering {file_name} -> {file_id} (Week {week_num})")
                        await conn.execute("""
                            INSERT INTO cls_file_mst (
                                file_id, cls_id, file_type_cd, file_nm, file_ext, 
                                file_format, upload_status, file_size, file_path, 
                                dwld_comp_dt, week_num, upload_type
                            ) VALUES ($1, $2, 'M', $3, $4, 'DOCUMENT', 'FU03', $5, $6, $7, $8, 'M')
                        """, file_id, cls_id, file_nm, file_ext, file_size, full_path, reg_dt, week_num)
                        status = 'FU03'

                    # Add to processing list if it needs embedding (FU03)
                    if status == 'FU03':
                        if cls_id not in cls_to_process:
                            cls_to_process[cls_id] = []
                        
                        cls_to_process[cls_id].append({
                            "file_name": f"{file_id}.{file_ext}",
                            "file_id": file_id,
                            "file_type_cd": 'M',
                            "file_path": full_path
                        })

        # 3. Trigger Embedding
        for cls_id, files_info in cls_to_process.items():
            if not files_info: continue
            
            # Fetch user_id again just in case (should be available from cls_info above)
            user_id = await conn.fetchval("SELECT user_id FROM cls_mst WHERE cls_id = $1", cls_id)
            
            print(f"\n>>> Starting embedding for Class: {cls_id} ({len(files_info)} files, User: {user_id})...")
            try:
                # Call the core embedding logic
                await process_and_embed_files(cls_id, files_info, user_id)
                print(f"Embedding completed for {cls_id}")
            except Exception as e:
                print(f"Error embedding files for {cls_id}: {e}")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(register_and_embed())
