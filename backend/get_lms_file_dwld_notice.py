import sys, os, asyncio, asyncpg, re, urllib.parse
import redis
import json
import requests
from bs4 import BeautifulSoup
from config import REDIS_CONFIG
from datetime import datetime, timezone, timedelta

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hallym_smart_lead.slead_member import SLeadMember
from hallym_smart_lead.slead_notice import get_notice_content
from config import DATABASE_CONFIG

# 설정
DEFAULT_BASE_PATH = "/data/storage/hlta_download_files"

redis_client = redis.Redis(**REDIS_CONFIG)

# 한국 시간대 설정 (UTC+9)
KST = timezone(timedelta(hours=9))

def get_current_time():
    """현재 한국 시간 반환"""
    return datetime.now(KST).replace(tzinfo=None)

async def get_db_pool():
    """데이터베이스 연결 풀 생성 - 타임존 설정 추가"""
    return await asyncpg.create_pool(
        **DATABASE_CONFIG,
        server_settings={
            'timezone': 'Asia/Seoul'
        }
    )

async def get_file_info(pool, file_id):
    """file_id로 파일 정보 조회"""
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM lms_file_notice WHERE file_id = $1", file_id)

def save_notice_content_as_txt(slead_member, notice_url, save_path, output_filename):
    """공지사항 URL의 내용만 텍스트 파일로 저장"""
    if not notice_url:
        return False, {"start_time": get_current_time()}
    
    try:
        # get_notice_content 함수를 사용하여 공지사항 내용 가져오기
        success, notice_data = get_notice_content(slead_member, notice_url)
        
        if not success:
            print(f"공지사항 내용 가져오기 실패: {notice_data}")
            return False, {"start_time": get_current_time()}
        
        # 파일 저장 경로 설정
        file_path = os.path.join(save_path, f"{output_filename}.txt")
        abs_file_path = os.path.abspath(file_path)
        
        # 저장 시간 기록
        start_time = get_current_time()
        
        # 공지사항 내용만 추출 (제목, 작성자, 날짜 제외)
        content_text = notice_data.get("content_text", "")
        
        # 텍스트 내용 저장 (내용만 저장)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_text)
        
        end_time = get_current_time()
        
        return True, {
            "file_path": abs_file_path,
            "file_ext": "txt",
            "file_size": os.path.getsize(file_path),
            "start_time": start_time,
            "end_time": end_time,
            "original_filename": f"{output_filename}.txt"
        }
    except Exception as e:
        print(f"공지사항 내용 저장 오류: {e}")
        return False, {"start_time": get_current_time()}

async def insert_file_info_to_db(pool, file_id, cls_id, file_path, file_size, start_time, end_time, file_name):
    """파일 정보를 DB에 저장"""
    try:
        async with pool.acquire() as conn:
            # 파일 타입 및 상태 설정
            file_type_cd = "N"
            file_format = "공지"
            file_ext = "txt"
            
            # 다운로드 상태 설정
            upload_status = "FU03"  # 성공
            dwld_comp_dt = end_time
            dwld_fail_dt = None
            
            # DB에 저장
            await conn.execute("""
            INSERT INTO cls_file_mst (
                file_id, cls_id, file_type_cd, file_ext, file_format, 
                upload_status, file_size, file_path, file_nm, week_num,
                dwld_start_dt, dwld_comp_dt, dwld_fail_dt,
                emb_start_dt, emb_comp_dt, emb_fail_dt, file_del_req_dt, emb_del_comp_dt
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT (file_id) DO UPDATE SET
                file_type_cd = $3, file_ext = $4, file_format = $5,
                upload_status = $6, file_size = $7, file_path = $8, file_nm = $9, week_num = $10,
                dwld_start_dt = $11, dwld_comp_dt = $12, dwld_fail_dt = $13
            """, file_id, cls_id, file_type_cd, file_ext, file_format,
                upload_status, file_size, file_path, file_name, None,
                start_time, dwld_comp_dt, dwld_fail_dt)
            
            return True
    except Exception as e:
        print(f"DB 저장 오류: {e}")
        return False

async def download_file_by_id(slead_member, pool, file_id, base_path, cls_id):   
    redis_hash_key = f"{cls_id}:material_status"

    # 파일 정보 조회
    file_info = await get_file_info(pool, file_id)
    if not file_info: return False
    
    # 파일 정보 추출
    notice_url = file_info.get("notice_url", "")
    notice_name = file_info.get("notice_nm", "")
    cls_id = file_info["cls_id"]
    
    # 파일명 정리 및 저장 경로 설정
    file_name = re.sub(r'[^\w\s\.\-가-힣]', '_', notice_name) if notice_name else f"file_{file_id}"
    save_path = os.path.abspath(os.path.join(base_path, cls_id, "notices"))
    os.makedirs(save_path, exist_ok=True)
    
    # 공지사항 저장
    print(f"공지사항 내용 저장: {file_id}, URL: {notice_url}")
    success, result = save_notice_content_as_txt(slead_member, notice_url, save_path, file_id)
    
    if not success:
        print(f"공지사항 내용 저장 실패: {file_id}")
        return False
    
    # 파일명 설정
    original_filename = notice_name
    if not original_filename or len(original_filename) < 3:
        original_filename = f"notice_{file_id}"
    
    # DB에 저장
    await insert_file_info_to_db(
        pool, file_id, cls_id, 
        result.get("file_path", ""), result.get("file_size", 0),
        result["start_time"], result.get("end_time"), original_filename
    )
    
    # Redis에 상태 정보 저장
    status_info = {
        "download_status": "completed",
        "stt_status": "completed",
        "embedding_status": "pending",            
        "file_ext": "txt",
        "file_nm": original_filename,
        "week_num": None
    }
            
    redis_client.hset(redis_hash_key, file_id, json.dumps(status_info))

    return (success, "txt")

async def download_multiple_files_n(user_id, user_pass, cls_id, file_ids, base_path=DEFAULT_BASE_PATH):
    # 로그인
    slead_member = SLeadMember()
    if not slead_member.login(user_id, user_pass):
        print("로그인 실패")
        return False

    # 비동기 DB 연결
    pool = await get_db_pool()
    stream_key = f"FDQ:{cls_id}:{user_id}"
    
    results = []
    try:
        for i, file_id in enumerate(file_ids, 1):
            print(f"[{i}/{len(file_ids)}] 다운로드 시작: {file_id}")
            try:
                success, file_ext = await download_file_by_id(slead_member, pool, file_id, base_path, cls_id)
                results.append((file_id, success, file_ext))
            except Exception as e:
                print(f"오류 발생: {e}")
                results.append((file_id, False, ""))
        
        final_message = {
            "task_type": "download",
            "status": "all_completed"
        }
        redis_client.xadd(stream_key, final_message, id='*')
        
        success_count = sum(1 for _, success, _ in results if success)
        return success_count == len(results)
    finally:
        await pool.close()

# 실행 부분
if __name__ == "__main__":
    file_ids = [
        "2024-2-17156-01-notice-38-f74038"
            ]
    asyncio.run(download_multiple_files_n(
        user_id="45932", 
        user_pass="abcde!234", 
        cls_id='2024-2-17156-01',
        file_ids=file_ids
    ))