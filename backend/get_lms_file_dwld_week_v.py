import sys, os, asyncio, asyncpg, subprocess, re, urllib.parse, time
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hallym_smart_lead.slead_member import SLeadMember
from config import DATABASE_CONFIG, REDIS_CONFIG
import redis
import json 
from dotenv import load_dotenv
import ffmpeg
from openai import OpenAI
# .env 파일에서 환경변수 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 설정
#DEFAULT_BASE_PATH = "./download_files"
#DEFAULT_LOCAL_MODEL_PATH = "./stt/whisper_models"
DEFAULT_BASE_PATH = "/data/storage/hlta_download_files"
DEFAULT_LOCAL_MODEL_PATH = "/app/tutor/stt/whisper_models"


redis_client = redis.Redis(**REDIS_CONFIG)

# 한국 시간대 설정 (UTC+9)
KST = timezone(timedelta(hours=9))

def get_current_time():
    """현재 한국 시간 반환"""
    return datetime.now(KST).replace(tzinfo=None)

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
            
            # UTC 대신 KST 시간 사용
            start_time = get_current_time()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            
            # UTC 대신 KST 시간 사용
            end_time = get_current_time()
            return True, {
                "file_path": abs_file_path,
                "file_ext": file_ext.lstrip('.'),
                "file_size": os.path.getsize(file_path),
                "start_time": start_time,
                "end_time": end_time,
                "original_filename": original_filename  # 확장자 포함된 원본 파일명
            }
        else:
            return False, {"start_time": get_current_time()}
    except:
        return False, {"start_time": get_current_time()}

def get_m3u8_url(slead_member, video_url):
    """동영상 페이지에서 m3u8 URL 추출"""
    # session = slead_member.get_session()
    session = slead_member
    if not session: return None
    
    try:
        response = session.get(video_url)
        if response.status_code != 200: return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # video 태그 또는 source 태그에서 찾기
        for video in soup.find_all('video'):
            src = video.get('src', '')
            if 'm3u8' in src: return src
            
            for source in video.find_all('source'):
                src = source.get('src', '')
                if 'm3u8' in src: return src
        
        return None
    except:
        return None

def download_video(video_url, save_path, output_filename):
    """yt-dlp로 동영상 다운로드"""
    try:
        os.makedirs(save_path, exist_ok=True)
        video_path = os.path.join(save_path, f"{output_filename}.mp4")
        abs_video_path = os.path.abspath(video_path)
        
        # UTC 대신 KST 시간 사용
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
            # UTC 대신 KST 시간 사용
            end_time = get_current_time()
            return True, {
                "file_path": abs_video_path,
                "file_ext": "mp4",
                "file_size": os.path.getsize(video_path),
                "start_time": start_time,
                "end_time": end_time,
                "original_filename": f"{output_filename}.mp4"
            }
        else:
            return False, {"start_time": start_time}
    except:
        # UTC 대신 KST 시간 사용
        return False, {"start_time": get_current_time()}

async def insert_file_info_to_db(pool, file_id, cls_id, file_type, file_ext, file_path, file_size, start_time, end_time, file_name, stt_start_dt=None, stt_comp_dt=None, stt_fail_dt=None, week_num=None):
    """파일 정보를 DB에 저장"""
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # 파일 타입 및 상태 설정 - file_ext로 확인
                file_type_cd = "V" if file_ext.lower() == "mp4" else "M"
                file_format = "영상" if file_ext.lower() == "mp4" else "문서"
                
                # 다운로드 상태 설정
                if end_time:
                    # 파일 확장자에 따른 upload_status 설정
                    allowed_extensions = {'pdf', 'txt', 'pptx', 'ppt', 'mp4'}
                    if file_ext.lower() in allowed_extensions:
                        upload_status = "FU03"  # 성공
                    else:
                        upload_status = "EP05"  # 허용되지 않는 확장자
                    dwld_comp_dt = end_time
                    dwld_fail_dt = None
                else:
                    upload_status = "FU04"  # 실패
                    dwld_comp_dt = None
                    dwld_fail_dt = get_current_time()
                
                # DB에 저장
                query = """
                INSERT INTO cls_file_mst (
                    file_id, cls_id, file_type_cd, file_ext, file_format, 
                    upload_status, file_size, file_path, file_nm,
                    dwld_start_dt, dwld_comp_dt, dwld_fail_dt,
                    stt_start_dt, stt_comp_dt, stt_fail_dt,
                    emb_start_dt, emb_comp_dt, emb_fail_dt, file_del_req_dt, emb_del_comp_dt,
                    week_num
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NULL, NULL, NULL, NULL, NULL, $16)
                ON CONFLICT (file_id) DO UPDATE SET
                    file_type_cd = $3, file_ext = $4, file_format = $5,
                    upload_status = $6, file_size = $7, file_path = $8, file_nm = $9,
                    dwld_start_dt = $10, dwld_comp_dt = $11, dwld_fail_dt = $12,
                    stt_start_dt = $13, stt_comp_dt = $14, stt_fail_dt = $15,
                    week_num = $16
                """
                
                await conn.execute(
                    query, 
                    file_id, cls_id, file_type_cd, file_ext, file_format,
                    upload_status, file_size, file_path, file_name,
                    start_time, dwld_comp_dt, dwld_fail_dt,
                    stt_start_dt, stt_comp_dt, stt_fail_dt,
                    week_num  # week_num 매개변수 사용
                )
                
                return True
    except Exception as e:
        print(f"DB 저장 오류: {e}")
        return False

async def update_stt_status(pool, file_id, stt_start_dt=None, stt_comp_dt=None, stt_fail_dt=None, stt_file_path=None):
    """STT 상태 업데이트"""
    try:
        async with pool.acquire() as conn:
            # 상태 코드 설정
            upload_status = None
            if stt_start_dt and not stt_comp_dt and not stt_fail_dt:
                upload_status = "FS02"  # STT 시작
            elif stt_comp_dt:
                upload_status = "FS03"  # STT 종료 (성공)
            elif stt_fail_dt:
                upload_status = "FS04"  # STT 실패
            
            # 쿼리 실행
            if upload_status:
                await conn.execute("""
                UPDATE cls_file_mst SET
                    stt_start_dt = $2,
                    stt_comp_dt = $3,
                    stt_fail_dt = $4,
                    stt_file_path = $5,
                    upload_status = $6
                WHERE file_id = $1
                """, file_id, stt_start_dt, stt_comp_dt, stt_fail_dt, stt_file_path, upload_status)
            else:
                await conn.execute("""
                UPDATE cls_file_mst SET
                    stt_start_dt = $2,
                    stt_comp_dt = $3,
                    stt_fail_dt = $4,
                    stt_file_path = $5
                WHERE file_id = $1
                """, file_id, stt_start_dt, stt_comp_dt, stt_fail_dt, stt_file_path)
            
            return True
    except Exception as e:
        print(f"STT 상태 업데이트 오류: {e}")
        return False

def convert_mp4_to_mp3(mp4_path: str, mp3_path: str):
    try:
        ffmpeg.input(mp4_path).output(
            mp3_path,
            ac=1, ar=16000, audio_bitrate='16k', vn=None
        ).run(overwrite_output=True)
        return True
    except Exception as e:
        print(f"MP4 to MP3 변환 오류: {e}")
        return False

def transcribe_video(video_path):
    try:
        start_time = time.time()

        mp3_path = os.path.splitext(video_path)[0] + ".mp3"
        if not convert_mp4_to_mp3(video_path, mp3_path):
            return False, None

        with open(mp3_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko",
                response_format="text"
            )

        txt_file_path = os.path.splitext(video_path)[0] + ".txt"
        with open(txt_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript)

        print(f"STT 완료: {os.path.basename(video_path)}")
        return True, txt_file_path
    except Exception as e:
        print(f"OpenAI STT 오류: {e}")
        return False, None

async def transcribe_video_async(pool, file_id, video_path):
    """동영상 파일을 텍스트로 변환 (비동기 래퍼)"""
    # STT 시작 시간 기록 및 상태 업데이트 (FS02)
    start_time = get_current_time()
    await update_stt_status(pool, file_id, stt_start_dt=start_time)  # 이 함수 호출로 upload_status가 FS02로 설정됨
    
    try:
        # 비동기 환경에서 동기 함수 실행을 위해 run_in_executor 사용
        loop = asyncio.get_event_loop()
        success, txt_file_path = await loop.run_in_executor(None, transcribe_video, video_path)
        
        if success and txt_file_path:
            # STT 성공 시간 기록 및 텍스트 파일 경로 저장 (FS03)
            comp_time = get_current_time()
            await update_stt_status(
                pool, file_id, 
                stt_start_dt=start_time, 
                stt_comp_dt=comp_time,
                stt_file_path=txt_file_path
            )  # 이 함수 호출로 upload_status가 FS03으로 설정됨
            return True
        else:
            # STT 실패 시간 기록 (FS04)
            fail_time = get_current_time()
            await update_stt_status(
                pool, file_id, 
                stt_start_dt=start_time, 
                stt_fail_dt=fail_time
            )  # 이 함수 호출로 upload_status가 FS04로 설정됨
            return False
    except Exception as e:
        print(f"STT 처리 중 오류: {e}")
        # STT 실패 시간 기록 (FS04)
        fail_time = get_current_time()
        await update_stt_status(
            pool, file_id, 
            stt_start_dt=start_time, 
            stt_fail_dt=fail_time
        )  # 이 함수 호출로 upload_status가 FS04로 설정됨
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
    week_num = file_info["week_num"]  # week_num 필드에서 직접 주차 정보 가져오기
    
    # VOD 파일만 처리하고 나머지는 건너뛰기
    if material_type != "vod":
        print(f"VOD가 아닌 파일 건너뜀: {file_id} - {material_name}")
        return True  # VOD가 아닌 파일은 건너뛰지만 성공으로 처리
    
    # VOD 다운로드 URL 준비
    if "mod/vod/view.php" in material_url:
        video_id = re.search(r'id=(\d+)', material_url)
        if video_id:
            material_url = f"https://smartlead.hallym.ac.kr/mod/vod/viewer.php?id={video_id.group(1)}"
    
    # 파일명 설정 (확장자 제외)
    original_filename = material_name
    if not original_filename or len(original_filename) < 3:
        original_filename = f"video_{file_id}"
    
    # m3u8 URL 가져오기
    m3u8_url = get_m3u8_url(slead_member, material_url)
    
    if m3u8_url:
        # 파일명 정리 및 저장 경로 설정 - 다운로드할 때만 폴더 생성
        material_name = re.sub(r'[^\w\s\.\-가-힣]', '_', material_name) if material_name else f"file_{file_id}"
        
        # 저장 경로를 base_path/cls_id/week_{week_num} 형태로 변경
        save_path = os.path.abspath(os.path.join(base_path, cls_id, f"week_{week_num}"))
        os.makedirs(save_path, exist_ok=True)
        
        # 동영상 다운로드
        success, result = download_video(m3u8_url, save_path, file_id)
        
        # DB에 저장 (STT 정보는 아직 없음)
        await insert_file_info_to_db(
            pool, file_id, cls_id, "vod", "mp4",
            result.get("file_path", ""), result.get("file_size", 0),
            result["start_time"], result.get("end_time"), original_filename,
            week_num=week_num
        )

        embedding_status = "pending"
        stt_status = "pending"
        download_status = "completed" if success else "failed"
            
        status_info = {
            "download_status": download_status,
            "stt_status": stt_status,
            "embedding_status": embedding_status,            
            "file_ext": "mp4",
            "file_nm" : original_filename,
            "week_num": week_num
        }
        print(f"[REDIS 저장] {redis_hash_key} → {file_id}: {json.dumps(status_info)}")
        redis_client.hset(redis_hash_key, file_id, json.dumps(status_info))
        
        # 다운로드 결과와 파일 경로 반환
        if success and result.get("file_path"):
            return True, result.get("file_path"), file_id
        else:
            return False, None, file_id
    else:
        # 다운로드 실패 시 DB에만 기록
        await insert_file_info_to_db(
            pool, file_id, cls_id, "vod", "mp4", "", 0, get_current_time(), None, original_filename,
            week_num=week_num
        )
        status_info = {
            "download_status": "failed",
            "stt_status": "failed",       
            "embedding_status": "failed",
            "file_ext": "mp4",
            "file_nm": original_filename,
            "week_num": week_num
        }
        redis_client.hset(redis_hash_key, file_id, json.dumps(status_info))
        return False, None, file_id

async def download_multiple_files_v(user_id, user_pass, cls_id, file_ids, base_path=DEFAULT_BASE_PATH):
    """여러 파일 다운로드 후 STT 처리"""
    print("download_multiple_files_v 함수 진입")
    # 로그인
    slead_member = SLeadMember()
    if not slead_member.login(user_id, user_pass):
        print("로그인 실패")
        return False
    
    # DB 연결 및 다운로드
    pool = await asyncpg.create_pool(
        min_size =10,
        max_size= 15,
        **DATABASE_CONFIG)
    try:
        # 1단계: 모든 파일 다운로드
        download_results = []
        vod_count = 0  # VOD 파일 카운트
        vod_success_count = 0  # 성공적으로 다운로드된 VOD 파일 카운트

        stream_key = f"FDQ:{cls_id}:{user_id}"        
        redis_hash_key = f"{cls_id}:material_status"

        for i, file_id in enumerate(file_ids, 1):
            print(f"[{i}/{len(file_ids)}] 처리 중: {file_id}")
            try:
                # 파일 정보 조회
                file_info = await get_file_info(pool, file_id)
                if not file_info:
                    print(f"파일 정보를 찾을 수 없음: {file_id}")
                    download_results.append((False, None, file_id, False))
                    continue
                
                # VOD 파일인지 확인
                material_type = file_info["material_type"]
                if material_type == "vod":
                    vod_count += 1
                    result = await download_file_by_id(slead_member, pool, file_id, base_path, cls_id)
                    if isinstance(result, tuple):
                        print(f"VOD 다운로드 시작: {file_id}")
                        success, file_path, file_id = result
                        if success and file_path:
                            vod_success_count += 1
                            print(f"VOD 다운로드 성공: {file_id}")
                            
                        else:
                            print(f"VOD 다운로드 실패: {file_id}")
                        download_results.append((success, file_path, file_id, True))
                    else:
                        download_results.append((result, None, file_id, True))
                else:
                    print(f"VOD가 아닌 파일 건너뜀: {file_id}")
                    download_results.append((True, None, file_id, False))  # VOD 아님 표시                
                
            except Exception as e:
                print(f"처리 오류: {e}")
                download_results.append((False, None, file_id, False))
        
        # 다운로드 결과 출력 (간소화)
        print(f"다운로드 결과: VOD {vod_count}개 중 {vod_success_count}개 성공")
        hash_info = redis_client.hgetall(redis_hash_key)        
        decoded_hash_info = {k.decode('utf-8'): json.loads(v.decode('utf-8')) for k, v in hash_info.items()}
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
        
        # 2단계: 다운로드 성공한 VOD 파일에 대해 STT 처리
        stt_results = []
        for success, file_path, file_id, is_vod in download_results:
            if success and file_path and is_vod:
                print(f"STT 처리 시작: {file_id}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "start"})

                try:
                    stt_success = await transcribe_video_async(pool, file_id, file_path)
                    stt_results.append((file_id, stt_success))
                    update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "completed"})
                    print(f"STT 처리 완료: {file_id} - {'성공' if stt_success else '실패'}")
                except Exception as e:
                    print(f"STT 처리 오류: {e}")                    
                    update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "failed"})
                    stt_results.append((file_id, False))
        
        # STT 결과 출력 (간소화)
        stt_success_count = sum(1 for _, success in stt_results if success)
        print(f"STT 처리 결과: {len(stt_results)}개 중 {stt_success_count}개 성공")
        
        return vod_success_count == vod_count
    finally:
        await pool.close()



def update_hash_and_notify_task(redis_hash_key: str, stream_key: str, task_type:str, file_id: str, update_fields: dict, extra_fields: dict = None):
    try:
        existing = redis_client.hget(redis_hash_key, file_id)
        status_info = {}
        if existing:
            status_info = json.loads(existing.decode('utf-8'))
        status_info.update(update_fields)
        redis_client.hset(redis_hash_key, file_id, json.dumps(status_info))
        message = {
            "task_type": task_type,
            "file_id": file_id,
            "hash_info": json.dumps(status_info)
        }
        if extra_fields:
            message.update(extra_fields)
        redis_client.xadd(stream_key, message, id='*')
    except Exception as e:
        print(f"Redis update error for {file_id}: {e}")

# 실행 부분
if __name__ == "__main__":

    file_ids = [
        "2024-2-17905-02-545223-b6777e",  # 테스트용 file_id (VOD)
        #"2024-2-17156-01-492077-707af0",  # 테스트용 file_id (일반 파일)
        #"2024-2-17156-01-490145-234bfa"
        #"2024-2-17905-02-547400-8ff55c"
        #"2024-2-17156-01-498492-1d5dae",
        #"2024-2-17156-01-503044-775943"
    ]
    asyncio.run(download_multiple_files_v(
        user_id="45932", 
        user_pass="abcde!234", 
        cls_id='2024-2-17156-01',
        file_ids=file_ids
    ))