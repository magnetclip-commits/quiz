import os
import json
import requests
import urllib3
from datetime import datetime

# Suppress InsecureRequestWarning for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration from Environment Variables
MILVUS_INGEST_URL = os.getenv("MILVUS_INGEST_URL")
TENANT_ID = os.getenv("MILVUS_TENANT_ID")
CALLBACK_URL_BASE = os.getenv("MILVUS_CALLBACK_URL")
X_API_KEY = os.getenv("MILVUS_API_KEY")

def construct_milvus_payload(file_record: dict):
    """
    Constructs the JSON payload for the Milvus Ingest API.
    file_record should contain:
    - file_id, cls_id, user_id, file_nm, file_ext, file_path, file_size, week_num, file_type_cd
    - stt_file_path (for videos)
    """
    file_id = file_record.get('file_id')
    cls_id = file_record.get('cls_id')
    user_id = file_record.get('user_id', 'system')
    file_nm = file_record.get('file_nm')
    file_ext = file_record.get('file_ext')
    file_path = file_record.get('file_path')
    stt_file_path = file_record.get('stt_file_path')
    file_size = file_record.get('file_size')
    week_num = file_record.get('week_num') or 1
    file_type_cd = file_record.get('file_type_cd')
    
    # Selection of content path: FOR VIDEOS, MUST USE stt_file_path. DO NOT FALLBACK.
    if file_type_cd == 'V':
        if not stt_file_path:
            raise ValueError(f"STT transcript path is missing for video {file_id}")
        final_file_path = stt_file_path
    else:
        final_file_path = file_path

    # PATH VALIDATION: Ensure the file is in a shared volume accessible by Milvus
    # Milvus JobManager typically only sees /data/storage
    if final_file_path and not final_file_path.startswith('/data/storage'):
        print(f"[{datetime.now()}] [WARNING] File {file_id} is in a non-shared path: {final_file_path}. Milvus may not be able to access it.")
    
    # Sanitize file name: remove newlines and leading/trailing spaces
    sanitized_file_nm = file_nm.replace('\n', ' ').replace('\r', '').strip()
    full_file_name = f"{sanitized_file_nm}.{file_ext}" if not sanitized_file_nm.endswith(file_ext) else sanitized_file_nm
    
    # Calculate actual file size of the file being sent (txt vs mp4)
    actual_size = file_size
    if final_file_path and os.path.exists(final_file_path):
        actual_size = os.path.getsize(final_file_path)
    
    # Mapping content type: M (Material) -> lecture_material, Others -> bulletin_board
    content_type = "lecture_material" if file_type_cd == 'M' else "bulletin_board"
    
    # Fixed S3 prefix as requested by admin
    S3_FIXED_PREFIX = "aiant"
    storage_url = f"{S3_FIXED_PREFIX}"
    
    return {
        "command": "ingest",
        "tenant_id": TENANT_ID,
        "user_id": user_id,
        "cls_id": cls_id,
        "file_id": file_id,
        "storage_url": storage_url,
        "metadata": {
            "files": [
                {
                    "filename": full_file_name,
                    "saved_pathname": final_file_path,
                    "size": actual_size
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
        },
        "callback": {
            "url": CALLBACK_URL_BASE,
            "id": file_id
        }
    }

def send_milvus_ingest(payload: dict):
    """Sends the payload to the Milvus Ingest API"""
    if not MILVUS_INGEST_URL:
        print("Error: MILVUS_INGEST_URL is not set.")
        return False, "MILVUS_INGEST_URL_NOT_SET"
        
    headers = {
        "Content-Type": "application/json",
        "x-api-key": X_API_KEY
    }
    
    try:
        # verify=False is used because JobManager might use a self-signed certificate
        resp = requests.post(MILVUS_INGEST_URL, headers=headers, json=payload, timeout=30.0, verify=False)
        return resp.status_code in [200, 201], resp.text
    except Exception as e:
        return False, str(e)
