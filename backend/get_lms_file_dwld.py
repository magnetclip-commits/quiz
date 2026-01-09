import sys, os, asyncio, asyncpg, subprocess, re, urllib.parse, hashlib
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hallym_smart_lead.slead_member import SLeadMember
from config import DATABASE_CONFIG

# 설정
DEFAULT_BASE_PATH = "./download_files"

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

def get_m3u8_url(slead_member, video_url):
    """동영상 페이지에서 m3u8 URL 추출"""
    
    # Selenium 브라우저 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 클라우드 환경에서 headless 모드 필수
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(video_url)

    # JavaScript 로딩 대기 (필요 시 조정)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # video 태그에서 찾기
    for video in soup.find_all("video"):
        src = video.get("src", "")
        if "m3u8" in src:
            return src

        for source in video.find_all("source"):
            src = source.get("src", "")
            if "m3u8" in src:
                return src

    # 스크립트 태그에서 찾기
    script_tags = soup.find_all("script")
    for script in script_tags:
        if script.string:
            urls = re.findall(r'(https?:\/\/[^\s]+\.m3u8)', script.string)
            if urls:
                return urls[0]

    return None

def download_video(video_url, save_path, output_filename):
    """yt-dlp로 동영상 다운로드"""
    try:
        os.makedirs(save_path, exist_ok=True)
        video_path = os.path.join(save_path, f"{output_filename}.mp4")
        abs_video_path = os.path.abspath(video_path)
        
        start_time = get_current_time()
        
        # yt-dlp 명령어 실행
        subprocess.run([
            "yt-dlp", "-o", video_path,
            "--no-check-certificate",
            "--referer", "https://smartlead.hallym.ac.kr/",
            "--quiet", video_url
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 파일 확인
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            return True, {
                "file_path": abs_video_path,
                "file_ext": "mp4",
                "file_size": os.path.getsize(video_path),
                "start_time": start_time,
                "end_time": get_current_time(),
                "original_filename": f"{output_filename}.mp4"
            }
        else:
            return False, {"start_time": start_time}
    except:
        return False, {"start_time": get_current_time()}

async def insert_file_info_to_db(pool, file_id, cls_id, file_type, file_ext, file_path, file_size, start_time, end_time, file_name):
    """파일 정보를 DB에 저장"""
    try:
        async with pool.acquire() as conn:
            # 파일 타입 및 상태 설정
            file_type_cd = "V" if file_type == "vod" else "M"
            file_format = "영상" if file_type == "vod" else "문서"
            
            # 다운로드 상태 설정
            if end_time:
                upload_status = "FU03"  # 성공
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
                upload_status, file_size, file_path, file_nm,
                dwld_start_dt, dwld_comp_dt, dwld_fail_dt,
                emb_start_dt, emb_comp_dt, emb_fail_dt, file_del_req_dt, emb_del_comp_dt
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT (file_id) DO UPDATE SET
                file_type_cd = $3, file_ext = $4, file_format = $5,
                upload_status = $6, file_size = $7, file_path = $8, file_nm = $9,
                dwld_comp_dt = $11, dwld_fail_dt = $12
            """, file_id, cls_id, file_type_cd, file_ext, file_format,
                upload_status, file_size, file_path, file_name,
                start_time, dwld_comp_dt, dwld_fail_dt)
            
            return True
    except Exception as e:
        print(f"DB 저장 오류: {e}")
        return False

async def download_file_by_id(slead_member, pool, file_id, base_path):
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
    
    # 파일명 정리 및 저장 경로 설정
    material_name = re.sub(r'[^\w\s\.\-가-힣]', '_', material_name) if material_name else f"file_{file_id}"
    save_path = os.path.abspath(os.path.join(base_path, cls_id, f"week_{week_num}"))
    os.makedirs(save_path, exist_ok=True)
    
    # 파일 타입에 따라 다운로드
    if material_type == "vod":
        # VOD 다운로드
        if "mod/vod/view.php" in material_url:
            video_id = re.search(r'id=(\d+)', material_url)
            if video_id:
                material_url = f"https://smartlead.hallym.ac.kr/mod/vod/viewer.php?id={video_id.group(1)}"
        
        # 파일명 설정 (확장자 제외)
        original_filename = material_name
        if not original_filename or len(original_filename) < 3:
            original_filename = f"video_{file_id}"
        
        # 파일 저장용 이름 (확장자 포함)
        full_filename = f"{original_filename}.mp4"
        
        # m3u8 URL 가져오기 및 다운로드
        m3u8_url = get_m3u8_url(slead_member, material_url)
        
        if m3u8_url:
            success, result = download_video(m3u8_url, save_path, file_id)
            await insert_file_info_to_db(
                pool, file_id, cls_id, "vod", "mp4",
                result.get("file_path", ""), result.get("file_size", 0),
                result["start_time"], result.get("end_time"), original_filename
            )
            return success
        else:
            await insert_file_info_to_db(
                pool, file_id, cls_id, "vod", "mp4", "", 0, get_current_time(), None, original_filename
            )
            return False
    else:
        # 일반 파일 다운로드
        success, result = download_file(slead_member, material_url, save_path, file_id)
        
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
            pool, file_id, cls_id, material_type, result.get("file_ext", ""),
            result.get("file_path", ""), result.get("file_size", 0),
            result["start_time"], result.get("end_time"), original_filename
        )
        return success

async def download_multiple_files(user_id, user_pass, file_ids, base_path=DEFAULT_BASE_PATH):
    """여러 파일 다운로드"""
    # 로그인
    slead_member = SLeadMember()
    if not slead_member.login(user_id, user_pass):
        print("로그인 실패")
        return False
    
    # DB 연결 및 다운로드 - 타임존 설정이 추가된 함수 사용
    pool = await get_db_pool()
    try:
        results = []
        for i, file_id in enumerate(file_ids, 1):
            print(f"[{i}/{len(file_ids)}] 다운로드: {file_id}")
            try:
                success = await download_file_by_id(slead_member, pool, file_id, base_path)
                results.append((file_id, success))
            except Exception as e:
                print(f"오류: {e}")
                results.append((file_id, False))
        
        success_count = sum(1 for _, success in results if success)
        print(f"총 {len(results)}개 중 {success_count}개 성공")
        return success_count == len(results)
    finally:
        await pool.close()

# 실행 부분
if __name__ == "__main__":
    file_ids = [
        "2024-2-17905-02-545223-b6777e",  # 테스트용 file_id (VOD)
        "2024-2-17156-01-492077-707af0",  # 테스트용 file_id (일반 파일)
        "2024-2-17156-01-490145-234bfa",
        "2024-2-17905-02-547400-8ff55c"
    ]
    
    asyncio.run(download_multiple_files(
        user_id="45932", 
        user_pass="abcde!234", 
        file_ids=file_ids
    ))