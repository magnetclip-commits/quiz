from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import DATABASE_CONFIG, REDIS_CONFIG
from fastapi import Request, UploadFile, File, HTTPException, Form, BackgroundTasks
import asyncpg
from embedding_m import process_and_embed_files, CHROMA_CLIENT
from remove_embed import delete_embedded_files
from typing import List, Tuple
import os
from pathlib import Path
from datetime import datetime
import hashlib
import asyncpg
import json
import unicodedata
import asyncio
import redis
import shutil
import os
from hallym_smart_lead.slead_member import SLeadMember
from get_lms_file_list import lms_file_list_main
from get_lms_notice_list import run_professor_test_async
# from get_lms_file_dwld_week_m import download_multiple_files_m
# from get_lms_file_dwld_week_v import download_multiple_files_v
# from embedding_m_0319_GH import process_new_files_and_update_summaries
# from embedding_v_0319_GH import process_new_videos_and_update_summaries
from remove_embed_m import delete_embedded_materials
from tasks import process_downloadM, process_downloadN, process_downloadV, process_external_video_stt
from utils.aes_util import decrypt
from get_lms_file_dwld_week_v import transcribe_video_async
from embedding_v import process_new_videos_and_update_summaries



from pydantic import BaseModel, Field

class ExternalFileInfo(BaseModel):
    orgFilename: str
    extName: str
    savedPathname: str
    size: int

class ExternalUploadRequest(BaseModel):
    cls_id: str
    user_id: str = "web_user"
    files: ExternalFileInfo
    files: ExternalFileInfo
    file_type_cd: str = "M" 
    move_to_storage: bool = False
    week_num: int = 1 


router = APIRouter()

# 외부 파일 업로드 및 임베딩 API
@router.post("/upload/external")
async def upload_external_files(request: ExternalUploadRequest):
    # 1. 파일 정보 추출
    file_info = request.files
    cls_id = request.cls_id
    user_id = request.user_id
    file_type_cd = request.file_type_cd
    week_num = request.week_num

    # 2. 파일 이름 구성
    file_nm = file_info.orgFilename
    if not file_nm.endswith(f".{file_info.extName}"):
         file_nm = f"{file_nm}.{file_info.extName}"

    # 3. 저장된 파일 경로
    file_path = file_info.savedPathname
    file_size = file_info.size
    current_time = datetime.now()
    
    # file_id 생성 (DB 트랜잭션 전에 필요할 경우를 대비해 미리 생성하거나, 트랜잭션 안에서 생성)
    # 여기서는 트랜잭션 안에서 생성하도록 유지하지만, move_to_storage 로직을 위해 미리 생성 필요할 수 있음
    # DB 트랜잭션 시작
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        async with conn.transaction():
            # generate_file_id 호출
            file_id = generate_file_id(cls_id, file_nm, current_time)

            # 파일 이동/복사 로직 (move_to_storage=True 인 경우)
            if request.move_to_storage:
                base_storage_dir = "/data/storage/edu"
                
                # 디렉토리 구조 결정 (notices 또는 week_N)
                if file_type_cd == 'N':
                    sub_dir = "notices"
                else:
                    sub_dir = f"week_{week_num}"
                    
                target_dir = os.path.join(base_storage_dir, cls_id, sub_dir)
                os.makedirs(target_dir, exist_ok=True)
                
                # 타겟 파일명: file_id.확장자
                target_filename = f"{file_id}.{file_info.extName}"
                target_path = os.path.join(target_dir, target_filename)
                
                print(f"Moving file from {file_path} to {target_path}")
                try:
                    # 안전한 이동 로직 (Safe Copy -> Verify -> Delete)
                    
                    # 1. 소스와 타겟이 동일한 파일인지 확인
                    if os.path.exists(target_path) and os.path.exists(file_path):
                         if os.path.samefile(file_path, target_path):
                            print(f"Source and destination are the same file: {file_path}")
                            # 이미 제자리에 있음 -> 아무 작업 안 함
                            pass
                         else:
                            # 2. 복사 (Copy)
                            shutil.copy2(file_path, target_path)
                            
                            # 3. 검증 (Verify)
                            if not os.path.exists(target_path):
                                raise Exception("Target file does not exist after copy.")
                            
                            src_size = os.path.getsize(file_path)
                            dst_size = os.path.getsize(target_path)
                            if src_size != dst_size:
                                raise Exception(f"Size mismatch after copy. Src: {src_size}, Dst: {dst_size}")
                                
                            # 4. 원본 삭제 (Delete)
                            os.remove(file_path)
                            print(f"Safe move completed: {file_path} -> {target_path}")
                            
                            file_path = target_path # DB 저장 경로 업데이트

                    else:
                        # 타겟이 없거나 소스가 없는 경우 (소스가 없으면 에러남)
                        # 여기서는 타겟 경로가 없는 경우를 가정하고 복사 수행
                        shutil.copy2(file_path, target_path)
                        
                        # 검증
                        if not os.path.exists(target_path):
                             raise Exception("Target file does not exist after copy.")
                        
                        # 삭제
                        os.remove(file_path)
                        file_path = target_path
                    
                    # 권한 설정 (선택사항)
                    if os.path.exists(file_path):
                        os.chmod(file_path, 0o644)

                except Exception as e:
                    print(f"Safe file move failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Safe file move failed: {str(e)}")

            
            # INSERT 쿼리 (CLS_FILE_MST)
            # dwld_comp_dt 등을 현재시각으로 설정하여 업로드 완료로 처리
            query = """
            INSERT INTO CLS_FILE_MST
            (FILE_ID, CLS_ID, FILE_TYPE_CD, FILE_NM, FILE_EXT,
            FILE_FORMAT, UPLOAD_STATUS, FILE_SIZE, FILE_PATH, week_num, 
            dwld_start_dt, dwld_comp_dt, upload_type)
            VALUES
            ($1, $2, $3, $4, $5,
            $6, 'FU03', $7, $8, $9, 
            CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul', CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul', 'USER')
            RETURNING FILE_ID;
            """
            
            # 한글 정규화
            normalized_filename = unicodedata.normalize("NFC", file_nm)

            # 중복 파일 확인 및 삭제 로직 추가 (Overwrite)
            check_query = """
                SELECT file_id, file_path FROM CLS_FILE_MST 
                WHERE cls_id = $1 AND file_nm = $2 AND upload_type = 'USER'
            """
            existing_record = await conn.fetchrow(check_query, cls_id, normalized_filename)
            
            if existing_record:
                existing_file_id = existing_record['file_id']
                existing_file_path = existing_record['file_path']
                print(f"Existing file found: {existing_file_id}. Overwriting...")
                
                # 1. DB에서 기존 레코드 삭제
                delete_query = "DELETE FROM CLS_FILE_MST WHERE file_id = $1"
                await conn.execute(delete_query, existing_file_id)
                
                # 2. ChromaDB에서 기존 임베딩 삭제
                try:
                    collection = CHROMA_CLIENT.get_collection(cls_id)
                    collection.delete(where={"file_id": existing_file_id})
                    print(f"Deleted embeddings for file_id: {existing_file_id}")
                except Exception as e:
                    print(f"Error deleting old embeddings: {e}")
                except Exception as e:
                    print(f"Error deleting old embeddings: {e}")
                    # 컬렉션이 없거나 삭제 실패해도 진행 (새 파일은 들어가야 하므로)
                
                # 3. 물리적 파일 삭제
                if existing_file_path and os.path.exists(existing_file_path):
                    try:
                        os.remove(existing_file_path)
                        print(f"Deleted old physical file: {existing_file_path}")
                    except Exception as e:
                        print(f"Failed to delete old physical file: {e}")

            # INSERT 실행
            await conn.fetchval(
                query,
                file_id,                        # $1
                cls_id,                         # $2
                file_type_cd,                   # $3
                normalized_filename,            # $4
                file_info.extName,              # $5
                f"application/{file_info.extName}", # $6
                file_size,                      # $7
                file_path,                      # $8
                week_num                        # $9
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        await conn.close()

    # 5. 임베딩 및 STT 트리거 (Celery Task 사용)
    result = None
    if file_type_cd == 'V':
        # 5-1. 동영상 STT 처리 (Celery Worker로 위임)
        process_external_video_stt.delay(file_id, file_path, cls_id, user_id)
        result = {"status": "processing", "detail": "STT and Embedding task dispatched to Celery worker"}
    else:
        # 문서 파일 임베딩 (기존 로직)
        files_info_for_embed = [{
            "file_name": normalized_filename,
            "file_id": file_id,
            "file_type_cd": file_type_cd,
            "file_path": file_path
        }]
    
        try:
            result = await process_and_embed_files(cls_id, files_info_for_embed, user_id)
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Embedding process error: {str(e)}")

    return {
        "message": "Upload and processing triggered successfully",
        "file_id": file_id,
        "result": result
    }



# ChromaDB 컬렉션 리스트 및 UUID 조회 API (순서 중요: {cls_id}보다 먼저 정의되어야 함)
@router.get("/debug/chroma/collections")
async def list_chroma_collections():
    try:
        collections = CHROMA_CLIENT.list_collections()
        result = {}
        for col in collections:
            result[col.name] = str(col.id)
            
        return {
            "message": "List of all ChromaDB collections",
            "count": len(collections),
            "collections": result  # { "TEST_CLASS_001": "uuid-..." }
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ChromaDB 확인용 임시 API
@router.get("/debug/chroma/{cls_id}")
async def inspect_chroma(cls_id: str):
    try:
        # 컬렉션 가져오기 (없으면 에러 발생 가능)
        collection = CHROMA_CLIENT.get_collection(cls_id)
        count = collection.count()
        # 데이터 살짝 엿보기 (최대 5개)
        peek = collection.peek(limit=5)
        
        return {
            "message": "ChromaDB Inspection Result",
            "collection_name": cls_id,
            "document_count": count,
            "metadatas": peek['metadatas'] # 메타데이터만 확인 (본문은 너무 길 수 있음)
        }
    except Exception as e:
        return {
            "status": "error", 
            "detail": f"Error accessing collection '{cls_id}': {str(e)}",
            "tip": "Collection might not exist if embedding failed completely."
        }


# ChromaDB 컬렉션 삭제 API
@router.delete("/debug/chroma/{cls_id}")
async def delete_chroma_collection(cls_id: str, check_dimension: bool = False):
    """
    지정된 cls_id에 해당하는 ChromaDB 컬렉션을 삭제합니다.
    check_dimension=True일 경우, 3072차원(text-embedding-3-large)이 아닌 경우에만 삭제합니다.
    """
    try:
        # 1. 컬렉션 삭제 시도
        collection_deleted = False
        try:
             # 컬렉션 존재 여부 확인 (있으면 삭제)
            try:
                collection = CHROMA_CLIENT.get_collection(cls_id)
                
                # 차원 확인 로직 (옵션)
                should_delete = True
                if check_dimension:
                    peek = collection.peek(limit=1)
                    embeddings = peek.get('embeddings')
                    if embeddings is not None and len(embeddings) > 0:
                        dim = len(embeddings[0])
                        if dim == 3072:
                            should_delete = False
                            print(f"Collection '{cls_id}' has dimension 3072. Skipping delete.")
                
                if should_delete:
                    CHROMA_CLIENT.delete_collection(cls_id)
                    collection_deleted = True
                
            except Exception:
                # 컬렉션이 없으면 이미 삭제된 것으로 간주하고 넘어감
                print(f"Collection '{cls_id}' not found. Skipping deletion.")
                pass
                
        except Exception as e:
            return {"status": "error", "detail": f"Error deleting collection: {str(e)}"}

        # 2. DB 상태 초기화 (재임베딩 가능하도록) - 컬렉션 삭제 여부와 상관없이 수행
        db_updated = 0
        try:
            conn = await asyncpg.connect(**DATABASE_CONFIG)
            try:
                 # upload_status를 FU03(업로드/다운로드 완료, 임베딩 대기)으로 초기화
                query = """
                    UPDATE cls_file_mst
                    SET upload_status = 'FU03',
                        emb_start_dt = NULL,
                        emb_comp_dt = NULL,
                        sumry_start_dt = NULL,
                        sumry_comp_dt = NULL
                    WHERE cls_id = $1
                """
                result = await conn.execute(query, cls_id)
                db_updated = int(result.split()[-1])
            finally:
                await conn.close()
        except Exception as db_e:
            return {
                "status": "partial_success",
                "message": f"Collection process finished but DB update failed: {str(db_e)}",
                "collection_deleted": collection_deleted
            }

        return {
            "status": "success",
            "message": f"Process completed. Collection deleted: {collection_deleted}, DB records reset: {db_updated}",
            "check_dimension": check_dimension
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


redis_client = redis.Redis(**REDIS_CONFIG)

'''
#파일 리스트 조회 API
@router.post("/filelist")
async def filelist(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_type_cd = body.get("file_type_cd")

    #삭제 완료 된 파일 데이터는 사용자에게 숨김
    query = """
        select *
        from file_mst cfm
        where cls_id = $1
            and file_type_cd = $2
            and upload_status != 'FD03';
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1, $2)에 값 매핑
            rows = await conn.fetch(query, cls_id, file_type_cd)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                     "cls_id": row["cls_id"],
                     "file_type_cd": row["file_type_cd"],
                     "file_nm": row["file_nm"],
                     "file_ext": row["file_ext"],
                     "file_format": row["file_format"],
                     "upload_status": row["upload_status"],
                     "file_reg_dt": row["file_reg_dt"],
                     "file_size": row["file_size"],
                     "file_path": row["file_path"]
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
'''
#파일 리스트 조회 API
@router.post("/rag/filelist")
async def rag_filelist(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_type_cd = body.get("file_type_cd")
    upload_type = body.get("upload_type")

    #삭제 완료 된 파일 데이터는 사용자에게 숨김
    query = """
        select *
        from cls_file_mst cfm
        where cls_id = $1
            and file_type_cd = $2
            and upload_status != 'FD03'
            and upload_type=$3;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1, $2)에 값 매핑
            rows = await conn.fetch(query, cls_id, file_type_cd, upload_type)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                     "cls_id": row["cls_id"],
                     "file_type_cd": row["file_type_cd"],
                     "file_nm": row["file_nm"],
                     "file_ext": row["file_ext"],
                     "file_format": row["file_format"],
                     "upload_status": row["upload_status"],
                     "file_reg_dt": row["dwld_comp_dt"],
                     "file_size": row["file_size"],
                     "file_path": row["file_path"],
                     "week_num": row["week_num"]
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
#LMS 수업 자료 리스트 가져오기 API
@router.post("/lmsfilelist")
async def lmsfilelist(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    year = body.get("year")
    semester = body.get("semester")
    cls_id = body.get("cls_id")

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)

        row = await conn.fetchrow("""
            SELECT user_pw
            FROM user_mst
            WHERE user_id = $1;
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        encrypted_pw = row["user_pw"]

        try:
            user_pw = decrypt(encrypted_pw)
        except Exception:
            user_pw = encrypted_pw

        await lms_file_list_main(
            input_user_id_pr = user_id, 
            input_user_pass_pr = user_pw, 
            year = year, 
            semester = semester, 
            cls_id = cls_id
            )

        return {"status": "success", "message": "LMS file list collection completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get LMS File list error: {str(e)}")
    finally:
        if conn:
            await conn.close()
    
#LMS 공지사항 리스트 가져오기 API
@router.post("/lmsnoticelist")
async def lmsnoticelist(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    # user_pw = body.get("user_pw")
    cls_id = body.get("cls_id")

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)

        row = await conn.fetchrow("""
            SELECT user_pw
            FROM user_mst
            WHERE user_id = $1;
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        encrypted_pw = row["user_pw"]

        try:
            user_pw = decrypt(encrypted_pw)
        except Exception:
            user_pw = encrypted_pw
    finally:
        if conn:
            await conn.close()

    try:
        await run_professor_test_async(
            user_id=user_id,
            user_pass=user_pw,
            save_to_db=True,
            target_cls_id=cls_id
            )
        
        return {"status": "success", "message": "LMS notice list collection completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get LMS Notice list error: {str(e)}")

#LMS 수업 자료 리스트 조회 API
@router.post("/materiallist")
async def materiallist(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    year = body.get("year")
    semester = body.get("semester")

    query = """
        SELECT a.file_id 
            ,a.cls_id
            ,a.user_id
            ,a.course_id
            ,a.week_num
            ,c.week_full_nm
            ,a.material_nm
            ,a.material_type
        FROM lms_file_weekly a
        INNER JOIN comm_smt_week c ON a.week_num = c.week_num AND c.yr =$1 AND c.smt = $2
        LEFT OUTER JOIN cls_file_mst f ON a.file_id =f.file_id
        WHERE material_type = ANY(ARRAY['ubfile', 'folder', 'vod'])
        AND a.cls_id = $3
        AND (
        	f.file_id IS NULL
        	OR ( 
         		f.upload_status = 'FD03'
    		)
    	)
        ORDER BY a.week_num, a.material_id
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1, $2)에 값 매핑
            # rows = await conn.fetch(query, user_id, cls_id)
            rows = await conn.fetch(query, year, semester, cls_id)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                    "cls_id": row["cls_id"],
                     "user_id": row["user_id"],
                     "course_id": row["course_id"],
                     "week_num": row["week_num"],
                     "week_full_nm": row["week_full_nm"],
                     "material_nm": row["material_nm"],
                     "material_type": row["material_type"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
#LMS 공지사항 리스트 조회 API
@router.post("/noticelist")
async def noticelist(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")

    query = """
        SELECT a.file_id 
                ,a.cls_id
                ,a.user_id
                ,a.course_id
                ,a.page_num
                ,a.notice_num
                ,a.notice_nm 
            FROM lms_file_notice a
            LEFT OUTER JOIN cls_file_mst f ON a.file_id =f.file_id
            WHERE a.cls_id = $1
            AND (
                f.file_id IS NULL
                OR ( 
                    f.upload_status = 'FD03'
                )
            )
            ORDER BY page_num , notice_num DESC
            ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1, $2)에 값 매핑
            rows = await conn.fetch(query, cls_id)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                    "cls_id": row["cls_id"],
                     "user_id": row["user_id"],
                     "course_id": row["course_id"],
                     "page_num": row["page_num"],
                     "notice_num": row["notice_num"],
                     "notice_nm": row["notice_nm"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    

@router.post("/materials/download")
async def materials_download(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id")
        # user_pw = body.get("user_pw")
        file_ids_m = body.get("file_ids_m")
        file_ids_v = body.get("file_ids_v")
        cls_id = body.get("cls_id")
        upload_type = (body.get("upload_type") or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"잘못된 요청: {str(e)}")

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)

        row = await conn.fetchrow("""
            SELECT user_pw
            FROM user_mst
            WHERE user_id = $1;
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        encrypted_pw = row["user_pw"]

        try:
            user_pw = decrypt(encrypted_pw)
        except Exception:
            user_pw = encrypted_pw
    finally:
        if conn:
            await conn.close()

    # 로그인은 upload_type이 명시되지 않은 경우에만 수행
    if not upload_type:
        slead_member = SLeadMember()
        if not slead_member.login(user_id, user_pw):
            print("로그인 실패")
            raise HTTPException(status_code=401, detail="로그인 실패")

    # 파일 ID 파싱 함수
    def parse_file_ids(file_ids):
        if isinstance(file_ids, str):
            try:
                parsed = json.loads(file_ids)
                if not isinstance(parsed, list):
                    return [parsed]
                else:
                    return parsed
            except json.JSONDecodeError:
                return [file_ids]
        elif not isinstance(file_ids, list):
            return [str(file_ids)]
        return file_ids

    # 각각의 파일 ID 리스트를 파싱
    file_ids_m = parse_file_ids(file_ids_m)
    file_ids_v = parse_file_ids(file_ids_v)

    # 모든 파일을 하나의 리스트로 결합 (초기 상태 등록용)
    combined_files = file_ids_m + file_ids_v

    print("Combined Files:", combined_files)

    redis_hash_key = f"{cls_id}:material_status"
    stream_key = f"FDQ:{cls_id}:{user_id}"

    set_status_standby = {
        "download_status": "pending",
        "embedding_status": "pending",
        "stt_status": "pending",
        "file_ext": "",
        "file_nm": "",  # 파일명 추가
        "week_num": "",  # 주차 정보 추가
    }

    combined_ids = []
    initial_state = {}

    for file_id in combined_files:
        try:
            fid = str(file_id)
            combined_ids.append(fid)
            # Redis 해시 저장
            redis_client.hset(redis_hash_key, fid, json.dumps(set_status_standby))
            # 초기 상태에 추가 (WebSocket과 동일한 구조로 만들기)
            initial_state[fid] = set_status_standby
        except Exception as e:
            print(f"Redis 해시 등록 오류 (file_id: {fid}): {e}")

    try:
        message_data = {
            "initial_state": json.dumps(initial_state)
        }
        redis_client.xadd(stream_key, message_data, id='*')
    except Exception as e:
        print(f"Redis 스트림 등록 오류: {e}")

    # M 파일용 스트림 메시지 등록
    if file_ids_m:
        try:
            message_data_M = {
                "user_id": user_id,
                "user_pw": user_pw,
                "file_ids": file_ids_m, # json.dumps 제거 (Celery는 dict 자체를 받아서 serialize 함)
                "cls_id": cls_id,
                "upload_type":upload_type
            }
            # redis_client.xadd("downloadM_queue", message_data_M, id='*')
            process_downloadM.delay(message_data_M)
            print("process_downloadM.delay 호출됨")
        except Exception as e:
            print(f"Celery Task(M) 호출 오류: {e}")
            raise HTTPException(status_code=500, detail=f"Celery Task(M) 호출 오류: {str(e)}")

    # V 파일용 스트림 메시지 등록
    if file_ids_v:
        try:
            message_data_V = {
                "user_id": user_id,
                "user_pw": user_pw,
                "file_ids": file_ids_v, # json.dumps 제거
                "cls_id": cls_id,
                "upload_type":upload_type
            }
            # redis_client.xadd("downloadV_queue", message_data_V, id='*')
            process_downloadV.delay(message_data_V)
            print("process_downloadV.delay 호출됨")
        except Exception as e:
            print(f"Celery Task(V) 호출 오류: {e}")
            raise HTTPException(status_code=500, detail=f"Celery Task(V) 호출 오류: {str(e)}")
        
    return {"message": "다운로드 및 임베딩이 시작되었습니다."}


@router.post("/notice/download")
async def notice_download(request: Request):
    try:
        body = await request.json()
        user_id = body.get("user_id")
        file_ids_n = body.get("file_ids_n")
        cls_id = body.get("cls_id")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"잘못된 요청: {str(e)}")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)

        row = await conn.fetchrow("""
            SELECT user_pw
            FROM user_mst
            WHERE user_id = $1;
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        
        encrypted_pw = row["user_pw"]

        try:
            user_pw = decrypt(encrypted_pw)
        except Exception:
            user_pw = encrypted_pw
    finally:
        if conn:
            await conn.close()

    slead_member = SLeadMember()
    if not slead_member.login(user_id, user_pw):
        print("로그인 실패")
        raise HTTPException(status_code=401, detail="로그인 실패")

    # 파일 ID 파싱 함수
    def parse_file_ids(file_ids):
        if isinstance(file_ids, str):
            try:
                parsed = json.loads(file_ids)
                if not isinstance(parsed, list):
                    return [parsed]
                else:
                    return parsed
            except json.JSONDecodeError:
                return [file_ids]
        elif not isinstance(file_ids, list):
            return [str(file_ids)]
        return file_ids

    file_ids_n = parse_file_ids(file_ids_n)

    if file_ids_n:
        try:
            message_data_N = {
                "user_id": user_id,
                "user_pw": user_pw,
                "file_ids": file_ids_n,
                "cls_id": cls_id
            }
            # redis_client.xadd("downloadN_queue", message_data_N, id='*')
            process_downloadN.delay(message_data_N)
            print("process_downloadN.delay 호출됨")
        except Exception as e:
            print(f"Celery Task(N) 호출 오류: {e}")
            raise HTTPException(status_code=500, detail=f"Celery Task(N) 호출 오류: {str(e)}")    

    return {"message": "다운로드 및 임베딩이 시작되었습니다."}


@router.post("/materials/dblist/M")
async def dblist_m(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_type_cd = body.get("file_type_cd")
    year = body.get("year")
    semester = body.get("semester")


    #삭제 완료된 것만 제외하고 리스트 조회
    query = """
        SELECT w.week_nm , w.week_full_nm ,ccm.cd_disp_nm, c.*
        FROM cls_file_mst c
        INNER JOIN lms_file_weekly l ON c.file_id = l.file_id  
        INNER JOIN comm_smt_week w ON l.week_num = w.week_num AND w.yr = $1 AND w.smt = $2
        INNER JOIN comm_cd_mst ccm ON ccm.cd_grp = 'UPLOAD_STATUS' AND c.upload_status = ccm.cd_nm 
        WHERE c.cls_id= $3
        AND file_type_cd = $4 --'M': 문서, 'V':동영상 
        AND upload_status != 'FD03'
        ORDER BY 1
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자에 값 매핑
            rows = await conn.fetch(query, year, semester, cls_id, file_type_cd)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                     "cls_id": row["cls_id"],
                     "cd_disp_nm" : row["cd_disp_nm"],
                     "file_type_cd": row["file_type_cd"],
                     "file_nm": row["file_nm"],
                     "file_ext": row["file_ext"],
                     "file_format": row["file_format"],
                     "upload_status": row["upload_status"],
                     "dwld_comp_dt": row["dwld_comp_dt"],
                     "file_size": row["file_size"],
                     "file_path": row["file_path"],
                     "week_nm": row["week_nm"],
                     "week_full_nm": row["week_full_nm"]
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@router.post("/materials/dblist/V")
async def dblist_v(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_type_cd = body.get("file_type_cd")
    year = body.get("year")
    semester = body.get("semester")


    #삭제 완료된 것만 제외하고 리스트 조회
    query = """
        SELECT w.week_nm , w.week_full_nm ,ccm.cd_disp_nm, c.*
        FROM cls_file_mst c
        INNER JOIN lms_file_weekly l ON c.file_id = l.file_id  
        INNER JOIN comm_smt_week w ON l.week_num = w.week_num AND w.yr = $1 AND w.smt = $2
        INNER JOIN comm_cd_mst ccm ON ccm.cd_grp = 'UPLOAD_STATUS' AND c.upload_status = ccm.cd_nm 
        WHERE c.cls_id= $3
        AND file_type_cd = $4 --'M': 문서, 'V':동영상 
        AND upload_status != 'FD03'
        ORDER BY 1
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자에 값 매핑
            rows = await conn.fetch(query, year, semester, cls_id, file_type_cd)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                     "cls_id": row["cls_id"],
                     "cd_disp_nm" : row["cd_disp_nm"],
                     "file_type_cd": row["file_type_cd"],
                     "file_nm": row["file_nm"],
                     "file_ext": row["file_ext"],
                     "file_format": row["file_format"],
                     "upload_status": row["upload_status"],
                     "dwld_comp_dt": row["dwld_comp_dt"],
                     "file_size": row["file_size"],
                     "file_path": row["file_path"],
                     "week_nm": row["week_nm"],
                     "week_full_nm": row["week_full_nm"]
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@router.post("/notice/dblist/N")
async def dblist_n(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_type_cd = body.get("file_type_cd")

    #삭제 완료된 것만 제외하고 리스트 조회
    query = """
        SELECT ccm.cd_disp_nm, c.*, n.notice_num
        FROM cls_file_mst c
        INNER JOIN lms_file_notice n ON c.file_id = n.file_id  
        INNER JOIN comm_cd_mst ccm ON ccm.cd_grp = 'UPLOAD_STATUS' AND c.upload_status = ccm.cd_nm 
        WHERE c.cls_id= $1
        AND upload_status != 'FD03'
        ORDER BY 1
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자에 값 매핑
            rows = await conn.fetch(query, cls_id)
            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"file_id": row["file_id"],
                     "cls_id": row["cls_id"],
                     "cd_disp_nm" : row["cd_disp_nm"],
                     "file_type_cd": row["file_type_cd"],
                     "file_nm": row["file_nm"],
                     "file_ext": row["file_ext"],
                     "file_format": row["file_format"],
                     "upload_status": row["upload_status"],
                     "dwld_comp_dt": row["dwld_comp_dt"],
                     "file_size": row["file_size"],
                     "file_path": row["file_path"],
                     "notice_num":row["notice_num"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# #파일 임베딩 API
# @router.post("/materials/embedding/M")
# async def materials_embedding(request: Request):
#     body = await request.json()
#     cls_id = body.get("cls_id")

#     try:
#         # 함수 인자로 notify_filestatus_toclients를 전달하여 순환 참조 해결
#         result = await process_new_files_and_update_summaries(cls_id, notify_filestatus_toclients)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Embedding process error: {str(e)}")

#     return {
#         "message": "모든 파일의 처리가 완료되었습니다.",
#         "result" : result
#     }

# #동영상 stt 완료된 파일 임베딩 API
# @router.post("/materials/embedding/V")
# async def materials_embedding_v(request: Request):
#     body = await request.json()
#     cls_id = body.get("cls_id")

#     try:
#         # 함수 인자로 notify_filestatus_toclients를 전달하여 순환 참조 해결
#         result = await process_new_videos_and_update_summaries(cls_id, notify_filestatus_toclients)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Embedding process error: {str(e)}")

#     return {
#         "message": "모든 파일의 처리가 완료되었습니다.",
#         "result" : result
#     }


#파일 삭제 API
@router.post("/materials/delete")
async def delete_materilas(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    file_ids = body.get("file_ids")

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            files_info = []
            allowed_exts = {"pdf", "txt", "ppt", "pptx", "mp4"}

            for file_id in file_ids:
                # 1. 파일 확장자 조회
                select_query = """
                SELECT FILE_ID, FILE_EXT, WEEK_NUM
                FROM CLS_FILE_MST
                WHERE FILE_ID = $1
                """
                row = await conn.fetchrow(select_query, file_id)

                if not row:
                    continue

                file_ext = row["file_ext"].lower()

                if file_ext not in allowed_exts:
                    # 허용되지 않은 확장자일 경우 상태 FD03으로 업데이트
                    update_query_invalid = """
                    UPDATE CLS_FILE_MST
                    SET UPLOAD_STATUS = 'FD03',
                        FILE_DEL_REQ_DT = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                    WHERE FILE_ID = $1
                    """
                    await conn.execute(update_query_invalid, file_id)
                    continue

                # 2. 유효한 확장자인 경우 상태 FD01로 업데이트
                update_query = """
                UPDATE CLS_FILE_MST
                SET UPLOAD_STATUS = 'FD01',
                    FILE_DEL_REQ_DT = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                WHERE FILE_ID = $1
                """
                await conn.execute(update_query, file_id)

                # 3. 파일 이름 및 정보 저장
                file_name = f"{row['file_id']}.{file_ext}"
                files_info.append({
                    "file_name": file_name,
                    "file_id": row['file_id'],
                    "week_num": f"week_{row['week_num']}"
                })

        finally:
            await conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    try:
        # ChromaDB 파일 삭제 함수 호출 (유효한 파일만 처리)
        # await delete_embedded_materials(cls_id, files_info)
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete process error: {str(e)}")

    return {
        "message": "모든 파일의 처리가 완료되었습니다.",
    }

#파일 업로드 디렉토리
UPLOADED_DIRECTORY = "/app/tutor/uploaded_files"
Path(UPLOADED_DIRECTORY).mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

def generate_file_id(cls_id: str, file_nm: str, file_reg_dt: datetime) -> str:
    """
    FILE_ID를 생성하는 함수
    형식: {cls_id}-{hash의 앞 6자리}
    hash는 cls_id, file_nm, file_reg_dt를 조합하여 생성
    """
    # 해시 생성을 위한 문자열 조합
    hash_string = f"{cls_id}{file_nm}{file_reg_dt.isoformat()}"

    # SHA-256 해시 생성
    hash_object = hashlib.sha256(hash_string.encode())
    hash_value = hash_object.hexdigest()[:6]  # 앞 6자리만 사용

    # 최종 FILE_ID 생성
    return f"{cls_id}-{hash_value}"

#파일 업로드 API
@router.post("/uploadfiles")
async def upload_files(files: List[UploadFile] = File(...), metadata: List[str] = Form(...)):
    # Metadata 파싱 및 INSERT 쿼리 생성
    try:
        parsed_metadata = [json.loads(data) for data in metadata]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON format: {str(e)}")

    # 파일과 메타데이터 매핑을 위한 리스트
    file_results = []

    # 데이터베이스 연결
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        async with conn.transaction():
            for file, meta in zip(files, parsed_metadata):

                # 현재 시간 생성 (file_reg_dt용)
                current_time = datetime.now()

                # FILE_ID 생성
                cls_id = meta.get("cls_id")
                file_nm = meta.get("file_nm")
                file_id = generate_file_id(cls_id, file_nm, current_time)

                # cls_id별 디렉토리 생성
                cls_directory = os.path.join(UPLOADED_DIRECTORY, str(cls_id))
                os.makedirs(cls_directory, exist_ok=True)

                # INSERT 쿼리 실행 및 생성된 FILE_ID 반환
                query = """
                INSERT INTO CLS_FILE_MST
                (FILE_ID, CLS_ID, FILE_TYPE_CD, FILE_NM, FILE_EXT,
                FILE_FORMAT, UPLOAD_STATUS, FILE_SIZE, FILE_PATH, week_num, dwld_start_dt, dwld_comp_dt,
                dwld_fail_dt, emb_start_dt, emb_comp_dt, emb_fail_dt, file_del_req_dt, emb_del_comp_dt, upload_type)
                VALUES
                ($1, $2, $3, $4, $5,
                $6, $7, $8, $9, 1, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul', CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul',
                NULL, NULL, NULL, NULL, NULL, NULL, 'USER')
                RETURNING FILE_ID;
                """
                # 파일 이름 정규화 (NFC)
                normalized_filename = unicodedata.normalize("NFC", meta.get("file_nm"))

                file_id = await conn.fetchval(
                    query,
                    file_id,                        #새로 생성한 FILE_ID 사용 $1
                    meta.get("cls_id"),                 #$2
                    meta.get("file_type_cd"),           #$3
                    normalized_filename,                #$4
                    meta.get("file_ext"),               #$5
                    meta.get("file_format"),            #$6
                    meta.get("upload_status", "FU01"),  #$7
                    meta.get("file_size"),              #$8
                    meta.get("file_path", "")           #$9
                )

                # 파일 저장
                file_ext = meta.get("file_ext", "").lstrip(".")
                # 파일 경로 생성
                file_path = os.path.join(cls_directory, f"{file_id}.{file_ext}")


                # 파일 내용 읽기 및 저장
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)

                # 파일 경로 업데이트
                update_query = """
                UPDATE CLS_FILE_MST
                SET UPLOAD_STATUS = 'FU03',
                FILE_PATH = $1,
                dwld_comp_dt = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                WHERE FILE_ID = $2
                """
                await conn.execute(update_query, str(file_path), file_id)

                file_results.append({
                    "filename": file.filename,
                    "file_id": file_id,
                    "saved_path": str(file_path)
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    finally:
        await conn.close()

    return {
        "message": "모든 파일의 처리가 완료되었습니다.",
        "files": file_results
    }

#파일 업로드 디렉토리
RAG_UPLOADED_DIRECTORY = "/app/tutor/download_files"
Path(RAG_UPLOADED_DIRECTORY).mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

@router.post("/rag/uploadfiles")
async def rag_upload_files(files: List[UploadFile] = File(...), metadata: List[str] = Form(...)):
    # Metadata 파싱 및 유효성 검사
    try:
        parsed_metadata = [json.loads(data) for data in metadata]
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON format: {str(e)}")

    # 파일 개수와 메타데이터 개수가 일치하는지 확인
    if len(files) != len(parsed_metadata):
        raise HTTPException(
            status_code=400, 
            detail=f"파일 개수({len(files)})와 메타데이터 개수({len(parsed_metadata)})가 일치하지 않습니다."
        )

    file_results = []
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    
    try:
        async with conn.transaction():
            # 프론트엔드에서 추가한 file_index로 정렬하여 순서 보장
            sorted_metadata = sorted(parsed_metadata, key=lambda x: x.get("file_index", 0))
            
            for file, meta in zip(files, sorted_metadata):
                # 현재 시간 생성 (file_reg_dt용)
                current_time = datetime.now()

                # FILE_ID 생성 - 기존 함수 사용
                cls_id = meta.get("cls_id")
                file_nm = meta.get("file_nm")
                file_id = generate_file_id(cls_id, file_nm, current_time)

                # week_num을 정수로 변환 
                week_num_str = meta.get("week_num", "1")
                try:
                    week_num = int(week_num_str)
                except (ValueError, TypeError):
                    week_num = 1  # 기본값

                # week_num별 디렉토리 생성
                week_directory = os.path.join(RAG_UPLOADED_DIRECTORY, str(cls_id), f"week_{week_num}")
                os.makedirs(week_directory, exist_ok=True)

                # 파일 이름 정규화 (NFC)
                normalized_filename = unicodedata.normalize("NFC", meta.get("file_nm"))

                # INSERT 쿼리 실행
                query = """
                INSERT INTO CLS_FILE_MST
                (FILE_ID, CLS_ID, FILE_TYPE_CD, FILE_NM, FILE_EXT,
                FILE_FORMAT, UPLOAD_STATUS, FILE_SIZE, FILE_PATH, week_num, dwld_start_dt, dwld_comp_dt,
                dwld_fail_dt, emb_start_dt, emb_comp_dt, emb_fail_dt, file_del_req_dt, emb_del_comp_dt, upload_type)
                VALUES
                ($1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul', CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul',
                NULL, NULL, NULL, NULL, NULL, NULL, 'USER')
                RETURNING FILE_ID;
                """

                returned_file_id = await conn.fetchval(
                    query,
                    file_id,                             # $1 - 새로 생성한 FILE_ID 사용
                    meta.get("cls_id"),                  # $2
                    meta.get("file_type_cd"),            # $3
                    normalized_filename,                 # $4
                    meta.get("file_ext"),                # $5
                    meta.get("file_format"),             # $6
                    meta.get("upload_status", "FU01"),   # $7
                    meta.get("file_size"),               # $8
                    meta.get("file_path", ""),           # $9
                    week_num                             # $10
                )

                # 파일 저장
                file_ext = meta.get("file_ext", "").lstrip(".")
                file_path = os.path.join(week_directory, f"{file_id}.{file_ext}")

                # 파일 내용 읽기 및 저장
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)

                # 파일 경로 업데이트
                update_query = """
                UPDATE CLS_FILE_MST
                SET UPLOAD_STATUS = 'FU03',
                FILE_PATH = $1,
                dwld_comp_dt = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                WHERE FILE_ID = $2
                """
                await conn.execute(update_query, str(file_path), file_id)

                file_results.append({
                    "filename": file.filename,
                    "file_id": file_id,
                    "saved_path": str(file_path)
                })

    except Exception as e:
        import logging
        logging.error(f"파일 업로드 오류: {str(e)}")
        logging.error(f"파일 개수: {len(files)}, 메타데이터 개수: {len(parsed_metadata)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    finally:
        await conn.close()

    return {
        "file_ids": [result["file_id"] for result in file_results]
    }


#파일 임베딩 API
@router.post("/embedding")
async def embedding_files(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    user_id = body.get("user_id", "web_user")

    #파일 업로드가 완료된 파일ID 리스트 불러오기 (임베딩 대기 상태인 FU03만 조회)
    query = """
        SELECT
            FILE_ID,
            FILE_EXT,
            FILE_TYPE_CD,
            FILE_PATH
        FROM CLS_FILE_MST
        WHERE CLS_ID = $1
        AND UPLOAD_STATUS = 'FU03'
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 쿼리 실행 및 결과 받기
            rows = await conn.fetch(query, cls_id)

            files_info = []

            for row in rows:
                file_name = f"{row['file_id']}.{row['file_ext']}"

                files_info.append({
                    "file_name": file_name,
                    "file_id": row['file_id'],
                    "file_type_cd": row['file_type_cd'],
                    "file_path": row['file_path']
                })
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    try:
        # 함수 인자로 user_id를 전달
        result = await process_and_embed_files(cls_id, files_info, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding process error: {str(e)}")

    return {
        "message": "모든 파일의 처리가 완료되었습니다.",
        "result" : result
    }


#파일 삭제 API
# @router.post("/deletefiles")
# async def delete_files(request: Request):
#     body = await request.json()
#     cls_id = body.get("cls_id")
#     file_ids = body.get("file_id")

'''
    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            files_info = []

            # 각 파일 ID에 대해 처리
            for file_id in file_ids:
                # 1. 파일 삭제 요청 시간 업데이트
                update_query = """
                UPDATE FILE_MST
                SET UPLOAD_STATUS = 'FD01',
                    FILE_DEL_REQ_DT = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                WHERE FILE_ID = $1
                """
                await conn.execute(update_query, file_id)

                # 2. 파일 확장자 조회
                select_query = """
                SELECT FILE_ID, FILE_EXT
                FROM FILE_MST
                WHERE FILE_ID = $1
                """
                row = await conn.fetchrow(select_query, file_id)

                if row:
                    # 3. file_name 생성 및 정보 저장
                    file_name = f"{row['file_id']}.{row['file_ext']}"
                    files_info.append({
                        "file_name": file_name,
                        "file_id": row['file_id']
                    })

        finally:
            await conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    try:
        # ChromaDB 파일 삭제 함수 호출
        # await delete_embedded_files(cls_id, files_info)
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete process error: {str(e)}")

    return {
        "message": "모든 파일의 처리가 완료되었습니다.",
    }
'''

# 웹소켓 연결을 관리하는 리스트
active_connections: List[Tuple[WebSocket, str]] = []  # (WebSocket, user_id) 튜플을 저장하는 리스트

@router.websocket("/ws/filestatus/{user_id}")
async def filestatus_websocket_endpoint(websocket: WebSocket, user_id: str):
    try:
        await websocket.accept()
        active_connections.append((websocket, user_id))
        
        while True:
            try:
                data = await websocket.receive_text()
                # 필요한 경우 수신된 메시지 처리
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        print(f"에러 발생: {str(e)}")
    finally:
        # 연결 종료 시 해당 사용자의 연결 정보 제거
        active_connections[:] = [conn for conn in active_connections if conn[0] != websocket]


@router.websocket("/ws/filestatus/stream/{user_id}/{cls_id}")
async def filestatus_websocket_endpoint2(websocket: WebSocket, user_id: str, cls_id: str):
    stream_key = f"FDQ:{cls_id}:{user_id}"
    redis_hash_key = f"{cls_id}:material_status"
    last_id = "$"

    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        hash_state = redis_client.hgetall(redis_hash_key)
        decoded_hash_state = {
            k.decode('utf-8'): json.loads(v.decode('utf-8')) for k, v in hash_state.items()
        }
        
        await websocket.send_text(json.dumps({"initial_state": decoded_hash_state}))
    except Exception as e:
        print(f"Error sending initial hash state for user {user_id}: {str(e)}")
    
    try:
        while True:
            # Redis 스트림에서 읽기
            result = await loop.run_in_executor(
                None,
                lambda: redis_client.xread({stream_key: last_id}, block=1000, count=1)
            )
            
            if result:
                for key, messages in result:
                    for message_id, message_data in messages:
                        last_id = message_id
                        # Redis 스트림 데이터 디코딩
                        decoded_data = {k.decode(): v.decode() for k, v in message_data.items()}
                        await websocket.send_text(json.dumps(decoded_data))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnect for user {user_id}")
    except Exception as e:
        print(f"Error in WebSocket endpoint for user {user_id}: {str(e)}")


async def notify_filestatus_toclients(message: str):
    """
    연결된 클라이언트에 메시지 전송
    """
    disconnected = []
    for websocket, _ in active_connections:  # 튜플에서 웹소켓 객체 추출
        try:
            await websocket.send_text(message)  # 웹소켓 객체로 직접 메시지 전송
        except Exception as e:
            print(f"메시지 전송 실패: {str(e)}")
            disconnected.append(websocket)
    
    # 연결이 끊긴 클라이언트 제거
    if disconnected:
        active_connections[:] = [conn for conn in active_connections if conn[0] not in disconnected]
