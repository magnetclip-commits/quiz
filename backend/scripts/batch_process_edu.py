import os
import requests
import json
import logging

# 설정
API_URL = "http://localhost:8085/file/upload/external"
BASE_DIR = "/data/storage/edu"
USER_ID = "admin_batch"
EXCLUDE_CLS_IDS = ["2025-20-003039-2-06"]

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file(cls_id, file_path):
    """단일 파일에 대해 API 호출"""
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lstrip('.') if ext else "bin"
        
        # 파일 타입 결정 (영상 V / 문서 M)
        video_exts = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        file_type_cd = "V" if ext.lower() in video_exts else "M"
        
        payload = {
            "cls_id": cls_id,
            "user_id": USER_ID,
            "file_type_cd": file_type_cd,
            "files": {
                "orgFilename": filename,
                "extName": ext,
                "savedPathname": os.path.abspath(file_path),
                "size": file_size
            }
        }

        logger.info(f"Processing: {filename} (Type: {file_type_cd}, Class: {cls_id})")
        
        # API 호출
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            result_data = res_json.get('result') or {}
            status = result_data.get('status', 'unknown')
            detail = result_data.get('detail', 'no detail')
            logger.info(f"  -> SUCCESS: {status} - {detail}")
        else:
            logger.error(f"  -> FALIED: Status {response.status_code}, Body: {response.text}")

    except Exception as e:
        logger.error(f"  -> ERROR: {file_path} 처리 중 오류: {e}")

def scan_and_process():
    """디렉토리 스캔 및 배체 처리"""
    if not os.path.exists(BASE_DIR):
        logger.error(f"Directory not found: {BASE_DIR}")
        return

    # 1. 과목(cls_id) 디렉토리 순회
    for cls_id in os.listdir(BASE_DIR):
        cls_dir = os.path.join(BASE_DIR, cls_id)
        if not os.path.isdir(cls_dir):
            continue

        if cls_id in EXCLUDE_CLS_IDS:
            logger.info(f"Skipping excluded class: {cls_id}")
            continue

        logger.info(f"=== Scanning Class: {cls_id} ===")
        
        # 2. 하위 디렉토리(week_*, notices) 순회 및 파일 찾기
        for root, dirs, files in os.walk(cls_dir):
            for file in files:
                # 숨김 파일 및 시스템 파일 제외
                if file.startswith('.') or file == "Thumbs.db":
                    continue
                
                file_path = os.path.join(root, file)
                process_file(cls_id, file_path)

if __name__ == "__main__":
    logger.info(f"Starting batch processing for files in {BASE_DIR}")
    scan_and_process()
