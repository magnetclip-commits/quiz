import asyncio
import sys
import os
import json
import requests 
import redis 
from celery import Celery
from celery.exceptions import MaxRetriesExceededError # 재시도 예외
import asyncpg

# --- 기존 임포트 및 설정 ---
from config import REDIS_CONFIG, DATABASE_CONFIG 
# from hallym_smart_lead.slead_member import SLeadMember
from get_lms_file_dwld_week_v import ( 
    download_file_by_id,
    transcribe_video_async,
    update_hash_and_notify_task,
    get_current_time,
    update_stt_status,
    download_multiple_files_v 
)

from get_lms_file_dwld_week_m import download_multiple_files_m
from embedding_v import process_new_videos_and_update_summaries
from embedding_m import process_new_files_and_update_summaries
from embedding_n import process_and_embed_new_files

from get_lms_file_dwld_notice import download_multiple_files_n

DEFAULT_BASE_PATH = "/data/storage/hlta_download_files"

# 폴더가 없으면 자동으로 생성해주는 안전 장치 (필수)
if not os.path.exists(DEFAULT_BASE_PATH):
    os.makedirs(DEFAULT_BASE_PATH, exist_ok=True)
    print(f"✅ 저장 디렉토리가 생성되었습니다: {DEFAULT_BASE_PATH}")
    
try:
    redis_client = redis.Redis(**REDIS_CONFIG)
    redis_client.ping() # 연결 테스트
    print("Redis 클라이언트 연결 성공")
except Exception as e:
    print(f"Redis 클라이언트 연결 실패: {e}")
    redis_client = None


# --- Celery 앱 설정 ---

# 1. 환경 변수에서 URL을 가져옵니다. 
# .env에 설정한 CELERY_BROKER_URL을 우선 사용하고, 없으면 기본값(localhost)을 사용합니다.
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# 2. Celery 인스턴스 생성
celery_app = Celery('tasks', broker=broker_url, backend=broker_url)

# --- (선택 사항) 일반 Redis 클라이언트 설정도 환경 변수화 ---
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", None)

try:
    # 패스워드가 있다면 포함하여 연결
    redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=0)
    redis_client.ping()
    print(f"✅ Redis 클라이언트 연결 성공 ({redis_host})")
except Exception as e:
    print(f"❌ Redis 클라이언트 연결 실패: {e}")
    redis_client = None

celery_app.conf.task_routes = {
    'task.transcribe_single_video': {'queue': 'stt_queue'},
    'task.process_embedding_v': {'queue': 'embedding_queue'}
}

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Seoul',
    enable_utc=False,
)

def run_coroutine(coro):
    """
    Celery 태스크 내에서 코루틴을 안전하게 실행합니다.
    이미 실행 중인 이벤트 루프를 사용하거나, 없을 경우 새 루프를 생성하여 실행합니다.
    """
    try:
        # 이미 실행 중인 루프가 있는지 확인
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 실행 중인 루프가 없으면 새로 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # loop.run_until_complete를 사용하여 코루틴 실행
    return loop.run_until_complete(coro)

async def get_db_pool():
    return await asyncpg.create_pool(**DATABASE_CONFIG)


def get_session_redis_key(user_id):
    return f"session:smartlead:{user_id}"

async def get_file_path_from_db(pool, file_id, cls_id):  ##### 파일 경로에 대해서 엘렉시와 조율 필요 emjay
    """
    데이터베이스(tb_file_sub)에서 file_id(strg_file_nm)를 이용해 파일 경로를 조회하는 함수
    SELECT strg_file_nm, strg_path_nm FROM tb_file_sub --- strg_path_nm; path
    """
    try:
        async with pool.acquire() as connection:
            # 파일 정보를 조회하는 쿼리 
            # query = """
            #     select file_path from cls_file_mst where file_id = $1;
            # """
            query = """
                select strg_path_nm from tb_file_sub where strg_file_nm = $1;
            """
            result = await connection.fetchrow(query, file_id)
            
            if result:
                # return result['file_path']
                return result['strg_path_nm']
            else:
                # print(f"데이터베이스에서 파일 정보를 찾을 수 없음: file_id={file_id}, cls_id={cls_id}")
                print(f"데이터베이스에서 파일 정보를 찾을 수 없음: strg_file_nm={file_id}, cls_id={cls_id}")
                return None
    except Exception as e:
        print(f"데이터베이스 파일 경로 조회 오류: {e}")
        return None
    
@celery_app.task(bind=True)
def trigger_download_tasks(self, message_data):
    print("[트리거] Celery 진입함", flush=True)
    user_id = message_data.get("user_id")
    user_pw = message_data.get("user_pw")
    file_ids = message_data.get("file_ids")
    cls_id = message_data.get("cls_id")
    upload_type = message_data.get("upload_type")

    if not redis_client:
        print("Redis 클라이언트 사용 불가. trigger_download_tasks 중단.")
        return {"status": "redis_error", "user_id": user_id, "error": "Redis client not available"}

    print(f"[Trigger Task] V 처리 시작 요청 - user: {user_id}, cls_id: {cls_id}, files: {len(file_ids)}, upload_type: {upload_type}")
    redis_session_key = get_session_redis_key(user_id)

    async def main_logic():
        pool = await get_db_pool()
                
        # upload_type이 있으면 다운로드를 건너뛰고 바로 STT 시작
        if upload_type:
            print(f"[Trigger Task] upload_type이 존재하므로 다운로드 건너뛰고 바로 STT 시작")
            try:
                task_results = []
                for file_id in file_ids:
                    # 데이터베이스에서 파일 정보를 조회하여 실제 파일 경로 찾기
                    video_path = await get_file_path_from_db(pool, file_id, cls_id)
                    print(f"[Trigger Task] 바로 STT 태스크 호출: {file_id}, path: {video_path}")
                    task = transcribe_single_video.delay(
                        cls_id,
                        user_id,
                        file_id,
                        video_path
                    )
                    task_results.append(task.id)
                return {"status": "stt_started", "user_id": user_id, "task_ids": task_results}
            except Exception as e:
                print(f"[Trigger Task] STT 직접 시작 중 오류 발생: {e}")
                return {"status": "error", "user_id": user_id, "error": str(e)}
            finally:
                if 'pool' in locals() and pool:
                    await pool.close()
        
        # upload_type이 없으면 기존 로직대로 다운로드 시작
        slead_member = SLeadMember()
        try:
            # 1. 로그인 시도
            if not slead_member.login(user_id, user_pw):
                 print(f"로그인 실패: {user_id}")
                 redis_client.delete(redis_session_key)
                 return {"status": "login_failed", "user_id": user_id}

            # 2. 세션 쿠키 추출
            session = slead_member.get_session()
            if not session:
                print(f"세션 객체를 얻을 수 없음: {user_id}")
                redis_client.delete(redis_session_key) 

            session_cookies = session.cookies.get_dict()

            try:
                if session_cookies:
                    redis_client.set(redis_session_key, json.dumps(session_cookies), ex=1800)
                    print(f"Redis에 세션 저장 완료: key={redis_session_key}")
                else:
                     print(f"경고: 추출된 세션 쿠키가 비어있습니다. user={user_id}")
                     redis_client.delete(redis_session_key)

            except Exception as redis_e:
                print(f"Redis 세션 저장 오류: {redis_e}")

            task_results = []
            for file_id in file_ids:
                print(f"[Trigger Task] VOD 파일 다운로드 태스크 호출: {file_id}")
                task = download_single_video.delay(
                    cls_id,
                    file_id,
                    DEFAULT_BASE_PATH,
                    user_id # 세션을 찾기 위한 user_id 전달
                )
                task_results.append(task.id)

        except Exception as e:
            print(f"[Trigger Task] 오류 발생: {e}")
            redis_client.delete(redis_session_key)
            return {"status": "error", "user_id": user_id, "error": str(e)}
        finally:
            if 'pool' in locals() and pool:
                await pool.close()

    return run_coroutine(main_logic())


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def download_single_video(self, cls_id, file_id, base_path, user_id):

    print(f"[Download Task] VOD 다운로드 시작: cls_id={cls_id}, file_id={file_id}, user_id={user_id}")
    stream_key = f"FDQ:{cls_id}:{user_id}"
    redis_hash_key = f"{cls_id}:material_status"
    redis_session_key = get_session_redis_key(user_id) # 세션 키 생성

    if not redis_client:
        print("Redis 클라이언트 사용 불가. download_single_video 중단.")
        update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "failed", "error": "Redis client not available"})
        return {"status": "redis_error", "file_id": file_id, "error": "Redis client not available"}

    async def main_logic():
        pool = await asyncpg.create_pool(**DATABASE_CONFIG)
        download_session = requests.Session() # 새 세션 객체 생성

        try:
            # 1. Redis에서 세션 정보 조회
            cookie_json = redis_client.get(redis_session_key)
            if not cookie_json:
                print(f"Redis에서 세션 정보를 찾을 수 없음 (만료 또는 없음): key={redis_session_key}")
                # 실패 상태 업데이트 및 종료
                update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "failed", "error": "Session not found or expired"})
                return {"status": "session_expired", "file_id": file_id}

            # 2. 세션 재구성
            try:
                loaded_cookies = json.loads(cookie_json)
                download_session.cookies.update(loaded_cookies)
                print(f"Redis에서 세션 로드 성공: key={redis_session_key}")
            except json.JSONDecodeError as json_e:
                print(f"Redis 세션 데이터 JSON 파싱 오류: {json_e}")
                redis_client.delete(redis_session_key)
                update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "failed", "error": "Invalid session data in Redis"})
                return {"status": "session_data_error", "file_id": file_id}

            update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "start"})
            success, file_path, returned_file_id = await download_file_by_id(download_session, pool, file_id, base_path, cls_id) 
            if success and file_path:
                print(f"[Download Task] VOD 다운로드 성공: {file_id}, path: {file_path}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "completed"})

                # STT 태스크 호출
                print(f"[Download Task] STT 태스크 호출: {file_id}")
                transcribe_single_video.delay(cls_id, user_id, file_id, file_path)
            else:
                print(f"[Download Task] VOD 다운로드 실패: {file_id}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "failed", "error": "Download failed"})
                return {"status": "download_failed", "file_id": file_id}

        except Exception as e:
            print(f"[Download Task] 오류 발생: {file_id}, error: {e}")
            update_hash_and_notify_task(redis_hash_key, stream_key, "download", file_id, {"download_status": "failed", "error": str(e)})
            try:
                self.retry(exc=e) # 재시도
            except MaxRetriesExceededError:
                 print(f"[Download Task] 최대 재시도 도달: {file_id}")
                 return {"status": "download_error_max_retry", "file_id": file_id, "error": str(e)}
            return {"status": "download_error", "file_id": file_id, "error": str(e)}
        finally:
            if 'pool' in locals() and pool:
                await pool.close()

    return run_coroutine(main_logic())


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def transcribe_single_video(self, cls_id, user_id, file_id, video_path):

    print(f"[STT Task] STT 처리 시작: cls_id={cls_id}, file_id={file_id}")
    stream_key = f"FDQ:{cls_id}:{user_id}"
    redis_hash_key = f"{cls_id}:material_status"

    if not redis_client: 
        print("Redis 클라이언트 사용 불가. transcribe_single_video 중단.")
        return {"status": "redis_error", "file_id": file_id, "error": "Redis client not available"}

    async def main_logic():
        pool = await get_db_pool()
        try:
            stt_success = await transcribe_video_async(pool, file_id, video_path)
            update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "start"})

            if stt_success:
                print(f"[STT Task] STT 처리 성공: {file_id}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "completed"})
                print(f"[STT Task][Queue: stt_queue] 임베딩 태스크 호출: {file_id} -> embedding_queue")
                process_embedding_v.apply_async(
                    args=[cls_id, user_id, file_id],
                    queue='embedding_queue' # 임베딩 전용 큐로 보냄
                )
            else:
                print(f"[STT Task] STT 처리 실패: {file_id}")
                return {"status": "stt_failed", "file_id": file_id}

        except Exception as e:
            print(f"[STT Task] 오류 발생: {file_id}, error: {e}")
            # 실패 시 DB 업데이트 및 Redis Hash 업데이트
            fail_time = get_current_time()
            try:
                # pool을 사용하여 DB 상태 업데이트
                 if pool:
                    await update_stt_status(pool, file_id, stt_fail_dt=fail_time)
                 else:
                    print("[STT Task] DB Pool 사용 불가로 상태 업데이트 실패")
            except Exception as db_err:
                print(f"[STT Task] DB STT 실패 상태 업데이트 오류: {db_err}")

            update_hash_and_notify_task(redis_hash_key, stream_key, "stt", file_id, {"stt_status": "failed", "error": str(e)})

            try:
                self.retry(exc=e)
            except MaxRetriesExceededError:
                print(f"[STT Task] 최대 재시도 도달: {file_id}")
                return {"status": "stt_error_max_retry", "file_id": file_id, "error": str(e)}
            return {"status": "stt_error", "file_id": file_id, "error": str(e)}
        finally:
            if 'pool' in locals() and pool:
                await pool.close()

    return run_coroutine(main_logic())

@celery_app.task(bind=True) # 필요시 재시도 등 추가
def process_embedding_v(self, cls_id, user_id, file_id):
    """
    (embedding_queue에서 실행)
    STT 완료 후 임베딩 로직을 수행합니다.
    """
    print(f"[Embedding Task][Queue: embedding_queue] 임베딩 처리 시작: cls_id={cls_id}, file_id={file_id}")
    stream_key = f"FDQ:{cls_id}:{user_id}"
    redis_hash_key = f"{cls_id}:material_status"

    if not redis_client:
        print("Redis 클라이언트 사용 불가. process_embedding_v 중단.")
        update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "failed", "error": "Redis client not available"})
        return {"status": "redis_error", "file_id": file_id, "error": "Redis client not available"}

    try:
        run_coroutine(process_new_videos_and_update_summaries(cls_id, user_id)) 
        print(f"[Embedding Task][Queue: embedding_queue] 임베딩 처리 완료: {file_id}")

    except Exception as e:
        print(f"[Embedding Task][Queue: embedding_queue] 임베딩 처리 오류: {file_id}, error: {e}")
 


@celery_app.task
def process_downloadM(message_data):
    user_id = message_data.get("user_id")
    user_pw = message_data.get("user_pw")
    file_ids = message_data.get("file_ids")
    cls_id = message_data.get("cls_id")
    upload_type = message_data.get("upload_type")
    
    if not upload_type:
        print(f"[Celery Task] M 다운로드 처리 시작 - user: {user_id}, file_ids: {file_ids}")
        run_coroutine(download_multiple_files_m(user_id, user_pw, cls_id, file_ids, DEFAULT_BASE_PATH))
        print(f"[Celery Task] M 다운로드 처리 완료 - user: {user_id}")
    
    
    print(f"[Celery Task] M 임베딩 처리 시작 - cls_id: {cls_id}")
    run_coroutine(process_new_files_and_update_summaries(cls_id, user_id))
    print(f"[Celery Task] M 임베딩 처리 완료 - user: {user_id}")

@celery_app.task
def process_downloadN(message_data):
    user_id = message_data.get("user_id")
    user_pw = message_data.get("user_pw")
    file_ids = message_data.get("file_ids")
    cls_id = message_data.get("cls_id")

    print(f"[Celery Task] N 다운로드 처리 시작 - user: {user_id}, file_ids: {file_ids}")
    run_coroutine(download_multiple_files_n(user_id, user_pw, cls_id, file_ids, DEFAULT_BASE_PATH))
    print(f"[Celery Task] N 다운로드 처리 완료 - user: {user_id}")
    
    
    print(f"[Celery Task] N 임베딩 처리 시작 - cls_id: {cls_id}")
    run_coroutine(process_and_embed_new_files(cls_id, user_id))
    print(f"[Celery Task] N 임베딩 처리 완료 - user: {user_id}")

@celery_app.task
def process_downloadV(message_data):
    user_id = message_data.get("user_id")
    user_pw = message_data.get("user_pw")
    file_ids = message_data.get("file_ids")
    cls_id = message_data.get("cls_id")
    upload_type = message_data.get("upload_type")

    print(f"[Celery Task] V 다운로드/STT 처리 시작 - user: {user_id}, file_ids: {file_ids}")
    # STT까지 포함된 다운로드 함수 호출
    run_coroutine(download_multiple_files_v(user_id, user_pw, cls_id, file_ids, DEFAULT_BASE_PATH))
    print(f"[Celery Task] V 다운로드/STT 처리 완료 - user: {user_id}")
    
    print(f"[Celery Task] V 임베딩 처리 시작 - cls_id: {cls_id}")
    run_coroutine(process_new_videos_and_update_summaries(cls_id, user_id))
    print(f"[Celery Task] V 임베딩 처리 완료 - user: {user_id}")

@celery_app.task
def process_external_video_stt(file_id, file_path, cls_id, user_id):
    """
    외부 업로드된 동영상 파일의 STT 및 임베딩 처리
    """
    print(f"[Celery Task] 외부 영상 처리 시작: {file_id}")
    
    async def logic():
        pool = await get_db_pool()
        try:
            print(f"[Celery Task] STT 시작: {file_id}")
            success = await transcribe_video_async(pool, file_id, file_path)
            
            if success:
                print(f"[Celery Task] STT 완료, 임베딩 시작: {file_id}")
                await process_new_videos_and_update_summaries(cls_id, user_id)
                print(f"[Celery Task] 외부 영상 처리 완료: {file_id}")
            else:
                print(f"[Celery Task] STT 실패: {file_id}")
        except Exception as e:
            print(f"[Celery Task] 외부 영상 처리 중 오류: {e}")
        finally:
            await pool.close()

    run_coroutine(logic())

