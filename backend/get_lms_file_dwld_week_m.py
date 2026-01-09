import sys, os, asyncio, asyncpg, re, urllib.parse
import redis
import json
from config import REDIS_CONFIG
from datetime import datetime, timezone, timedelta

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from hallym_smart_lead.hallym_smart_lead.slead_member import SLeadMember
from hallym_smart_lead.slead_member import SLeadMember
from config import DATABASE_CONFIG

# 설정
#DEFAULT_BASE_PATH = "./download_files"
DEFAULT_BASE_PATH = "/app/tutor/download_files"

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
        return await conn.fetchrow("SELECT * FROM lms_file_weekly WHERE file_id = $1", file_id)

def download_file(slead_member, url, save_path, output_filename):
    """URL에서 파일 다운로드"""
    session = slead_member.get_session()
    if not session:
        return False, {"start_time": get_current_time()}
    
    try:
        # URL 파싱 및 다운로드 URL 구성
        parsed_url = urllib.parse.urlparse(url)
        file_ext = os.path.splitext(parsed_url.path)[1] or ".bin"
        
        query_params = urllib.parse.parse_qs(parsed_url.query)
        query_params['forcedownload'] = ['1']
        download_url = urllib.parse.urlunparse((
            parsed_url.scheme, parsed_url.netloc, parsed_url.path,
            parsed_url.params, urllib.parse.urlencode(query_params, doseq=True), parsed_url.fragment
        ))
        
        # 파일 다운로드
        response = session.get(download_url, stream=True, timeout=30)
        
        if response.status_code == 200:
            # 파일명 추출 및 저장
            original_filename = os.path.basename(parsed_url.path)
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition and 'filename=' in content_disposition:
                filename_part = content_disposition.split('filename=')[1]
                original_filename = filename_part.split('"')[1] if '"' in filename_part else filename_part.split(';')[0].strip()
                original_filename = urllib.parse.unquote(original_filename)
                original_filename = re.sub(r'[^\w\s\.\-가-힣]', '_', original_filename)
                file_ext = os.path.splitext(original_filename)[1] or file_ext
            
            # 파일 저장
            output_file = f"{output_filename}{file_ext}"
            file_path = os.path.join(save_path, output_file)
            abs_file_path = os.path.abspath(file_path)
            
            start_time = get_current_time()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            
            return True, {
                "file_path": abs_file_path,
                "file_ext": file_ext.lstrip('.'),
                "file_size": os.path.getsize(file_path),
                "start_time": start_time,
                "end_time": get_current_time(),
                "original_filename": original_filename  # 확장자 포함된 원본 파일명
            }
        else:
            return False, {"start_time": get_current_time()}
    except:
        return False, {"start_time": get_current_time()}

async def insert_file_info_to_db(pool, file_id, cls_id, file_type, file_ext, file_path, file_size, start_time, end_time, file_name, file_info):
    """파일 정보를 DB에 저장"""
    try:
        async with pool.acquire() as conn:
            # 파일 타입 및 상태 설정 - file_ext로 확인
            file_type_cd = "V" if file_ext.lower() == "mp4" else "M"
            file_format = "영상" if file_ext.lower() == "mp4" else "문서"
            
            # 허용된 확장자 목록
            allowed_extensions = {'pdf', 'txt', 'pptx', 'ppt', 'mp4'}
            
            # 다운로드 상태 설정
            if end_time:
                if file_ext.lower() in allowed_extensions:
                    upload_status = "FU03"  # 성공
                else:
                    upload_status = "EP05"  # 허용되지 않은 확장자
                dwld_comp_dt = end_time
                dwld_fail_dt = None
            else:
                upload_status = "FU04"  # 실패
                dwld_comp_dt = None
                dwld_fail_dt = get_current_time()
            
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
                upload_status, file_size, file_path, file_name, file_info["week_num"],
                start_time, dwld_comp_dt, dwld_fail_dt)
            
            return True
    except Exception as e:
        print(f"DB 저장 오류: {e}")
        return False

async def download_file_by_id(slead_member, pool, file_id, base_path, cls_id):   

    redis_hash_key = f"{cls_id}:material_status"

    """file_id로 파일 다운로드"""
    # 파일 정보 조회
    file_info = await get_file_info(pool, file_id)
    if not file_info: return False
    
    # 파일 정보 추출
    material_url = file_info["material_url"]
    material_type = file_info["material_type"]
    material_name = file_info.get("material_nm", "")
    cls_id = file_info["cls_id"]
    week_num = file_info["week_num"]
    
    # VOD 파일은 건너뛰기
    if material_type == "vod":
        print(f"VOD 파일 건너뜀: {file_id} - {material_name}")
        return True  # VOD 파일은 건너뛰지만 성공으로 처리
    
    # 파일명 정리 및 저장 경로 설정
    material_name = re.sub(r'[^\w\s\.\-가-힣]', '_', material_name) if material_name else f"file_{file_id}"
    save_path = os.path.abspath(os.path.join(base_path, cls_id, f"week_{week_num}"))
    os.makedirs(save_path, exist_ok=True)
    
    # 일반 파일 다운로드
    success, result = download_file(slead_member, material_url, save_path, file_id)

    file_ext = result.get("file_ext", "")
    
    # 파일명 설정 (확장자 제외)
    original_filename = material_name
    if not original_filename or len(original_filename) < 3:
        # 다운로드 결과에서 파일명 가져오기
        downloaded_filename = result.get("original_filename", f"file_{file_id}")
        # 확장자 제거
        original_filename = os.path.splitext(downloaded_filename)[0]
    elif '.' in original_filename:
        # 확장자가 있으면 제거
        original_filename = os.path.splitext(original_filename)[0]
    
    # DB에 저장
    await insert_file_info_to_db(
        pool, file_id, cls_id, material_type, file_ext,
        result.get("file_path", ""), result.get("file_size", 0),
        result["start_time"], result.get("end_time"), original_filename, file_info
    )

    allowed_extensions = ['pdf', 'txt', 'pptx', 'ppt', 'mp4']
            
     # 각 파일에 대한 상태 결정
    if file_ext not in allowed_extensions:
        embedding_status = "deny"
        stt_status = "deny"
    else:
        embedding_status = "pending"
        stt_status = "completed"
    
    download_status = "completed" if success else "failed"
            
    status_info = {
        "download_status": download_status,
        "stt_status": stt_status,
        "embedding_status": embedding_status,            
        "file_ext": file_ext,
        "file_nm" : original_filename,
        "week_num": week_num
    }
            
    redis_client.hset(redis_hash_key, file_id, json.dumps(status_info))


    return (success, file_ext)

async def download_multiple_files_m(user_id, user_pass, cls_id, file_ids, base_path=DEFAULT_BASE_PATH):
    # 로그인
    slead_member = SLeadMember()
    if not slead_member.login(user_id, user_pass):
        print("로그인 실패")
        return False

    # 비동기 DB 연결 (타임존 설정 포함)
    pool = await get_db_pool()

    stream_key = f"FDQ:{cls_id}:{user_id}"
    redis_hash_key = f"{cls_id}:material_status"
    
    
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
        
        hash_info = redis_client.hgetall(redis_hash_key)
        decoded_hash_info = {k.decode('utf-8'): json.loads(v.decode('utf-8')) for k, v in hash_info.items()}
        # 최종 메시지 생성 및 xadd
        final_message = {
            "task_type": "download",
            "status": "all_completed",
            "hash_info": json.dumps(decoded_hash_info)
        }
        redis_client.xadd(stream_key, final_message, id='*')

        for file_id, status_info in list(decoded_hash_info.items()):
            if status_info.get("embedding_status") == "deny":
                redis_client.hdel(redis_hash_key, file_id)
            elif status_info.get("download_status") == "failed":
                redis_client.hdel(redis_hash_key, file_id)

        
        success_count = sum(1 for _, success, _ in results if success)
        return success_count == len(results)
    finally:
        await pool.close()



# 실행 부분
if __name__ == "__main__":
    file_ids = [
        # "2024-2-17905-02-545223-b6777e",  # 테스트용 file_id (VOD)
         "2024-2-17156-01-492077-707af0",  # 테스트용 file_id (일반 파일)
        # "2024-2-17156-01-490145-234bfa",
        # "2024-2-17905-02-547400-8ff55c"
        #'2024-2-17156-01-493355-358b44' #url링크 
            ]
    
    asyncio.run(download_multiple_files_m(
        user_id="45932", 
        user_pass="abcde!234", 
        cls_id='2024-2-17156-01',
        file_ids=file_ids
    ))