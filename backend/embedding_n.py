'''
@modified: embedding_n_0414_DH
2025.08.14 chromadb 서버방식으로 변경
'''
import asyncio # 표준 라이브러리 임포트
import os
import subprocess
import asyncpg # 써드파티 라이브러리 임포트
import nltk
from dotenv import load_dotenv
from fastapi import HTTPException
from pptx import Presentation
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from config import DATABASE_CONFIG,REDIS_CONFIG # 로컬 모듈 임포트
from datetime import datetime
import json  # JSON 처리를 위해 추가
import redis

nltk.download('punkt')
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

redis_client = redis.Redis(**REDIS_CONFIG)

# 디렉토리 권한 확인 및 설정 함수
def ensure_directory_permissions(path, mode=0o775):
    """
    디렉토리 및 상위 디렉토리의 권한을 확인하고 설정합니다.
    """
    try:
        # 디렉토리가 이미 존재하는 경우
        if os.path.exists(path):
            current_mode = os.stat(path).st_mode & 0o777
            if current_mode != mode:
                print(f"권한 변경 필요: {path} (현재: {current_mode:o}, 목표: {mode:o})")
                os.chmod(path, mode)
                print(f"권한 변경 완료: {path}")
            else:
                print(f"권한 설정 정상: {path}")
            return
            
        # 디렉토리가 존재하지 않으면 상위 디렉토리 권한 확인
        parent = os.path.dirname(path)
        if parent and parent != path:  # 무한 루프 방지
            ensure_directory_permissions(parent, mode)
            
        # 디렉토리 생성 후 권한 설정
        os.makedirs(path, exist_ok=True)
        os.chmod(path, mode)
        print(f"디렉토리 생성 및 권한 설정 완료: {path}")
    except Exception as e:
        print(f"권한 설정 중 오류 발생: {path} - {e}")
        import traceback
        print(traceback.format_exc())

async def get_file_name_from_db(file_id: str) -> str:
    query = """
        SELECT file_nm
        FROM cls_file_mst
        WHERE file_id = $1
    """
    
    conn = None
    try:
        print(f"Fetching file name for file_id: {file_id}")  # 디버깅 로그
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        result = await conn.fetchval(query, file_id)
        print(f"Query result for file_id {file_id}: {result}")  # 디버깅 로그
        return result
    except Exception as e:
        print(f"Error fetching file name from DB: {e}")  # 디버깅 로그
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

def preprocess_text_with_metadata(text: str, metadata: dict) -> str:
    """
    텍스트에 메타데이터 정보를 추가
    """
    file_type = metadata.get('종류', '알 수 없음')
    title = metadata.get('title', '알 수 없음')
    
    prefix = f"이 문서는 교수자가 [{file_type}] 카테고리에 업로드한 내용입니다. "
    prefix += f"참고한 파일의 제목은 [{title}]입니다."
    
    # 텍스트가 비어있거나 "(내용 없음)"만 포함된 경우
    if not text.strip() or text.strip() == "(내용 없음)":
        # 제목에서 의미있는 정보 추출
        title_info = title.replace("(내용 없음)", "").strip()
        prefix += f"\n\n이 {file_type}의 주요 내용은 제목에 포함되어 있습니다: {title_info}"
    else:
        prefix += f"\n\n{text}"
    
    return prefix

def load_text_file(file_path, title=None, file_type_str=None):
    """
    텍스트 파일 로드 및 처리
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # 빈 내용이거나 "(내용 없음)"인 경우 기본 문서 생성
    if not text or text == "(내용 없음)":
        # 제목에서 의미있는 정보 추출
        title_info = title.replace("(내용 없음)", "").strip() if title else ""
        text = f"이 {file_type_str}은 제목에 주요 내용이 포함되어 있습니다: {title_info}"
    
    doc = Document(
        page_content=text,
        metadata={
            "title": title if title else "",
            "종류": file_type_str if file_type_str else "",
        }
    )
    return [doc]

async def load_file(file_path, file_type_cd, file_id):
    """
    파일을 로드하고 태그를 추가하는 함수
    """
    type_mapping = {
        "S": "강의일정",
        "N": "공지사항",
        "M": "수업자료",
        "V": "강의영상"
    }
    file_type_str = type_mapping.get(file_type_cd, "기타")
    
    # DB에서 파일명 가져오기
    title = await get_file_name_from_db(file_id)
    if not title:
        print(f"Database did not return a title for file_id {file_id}. Using file name from path.")
        title = os.path.basename(file_path)
    
    print(f"Loading file with title: {title} and type: {file_type_str}")  # 디버깅 로그 추가

    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".txt":
        documents = load_text_file(file_path, title, file_type_str)
    else:
        raise ValueError(f"FILE_FORMAT_ERROR: 지원되지 않는 파일 형식입니다: {ext}")

    return documents

async def update_status_in_db(status, dt_tag, file_id):
    """
    데이터베이스에서 파일 상태 업데이트
    """
    if dt_tag == 'NO_UPDATE':
        query = """
            UPDATE cls_file_mst
            SET upload_status = $1
            WHERE FILE_ID = $2
        """
    else:
        query = f"""
            UPDATE cls_file_mst
            SET upload_status = $1,
            {dt_tag} = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
            WHERE FILE_ID = $2
        """

    conn = None
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        await conn.execute(query, status, file_id)
        print(f"DB 업데이트: {file_id} - {status}")
    except Exception as e:
        print(f"DB 업데이트 오류: {file_id} - {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

async def identify_new_files(cls_id: str) -> list:
    """
    임베딩되지 않은 새로운 파일들을 식별
    """
    query = """
        SELECT 
            file_id,
            cls_id,
            file_type_cd,
            file_nm,
            file_path,
            upload_status
        FROM cls_file_mst
        WHERE cls_id = $1
        AND file_type_cd = 'N'
        AND upload_status IN ('FU03')  -- 파일 업로드 완료인 파일 찾기
    """
            # AND file_del_req_dt IS NULL
        # AND emb_del_comp_dt IS NULL
    #         AND upload_status IN ('FU03', 'EP04')  -- 파일 업로드 완료이거나 임베딩 처리 실패

    conn = None
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        files = await conn.fetch(query, cls_id)
        return [dict(file) for file in files]
    except Exception as e:
        print(f"Error identifying new files: {e}")
        return []
    finally:
        if conn:
            await conn.close()

async def process_and_embed_files(cls_id: str, files_info: list, user_id: str):
    """
    파일 처리 및 임베딩 수행
    """
    base_path = f"/app/tutor/download_files/{cls_id}"

    # 디렉토리 권한 확인 및 설정
    ensure_directory_permissions(base_path)
    
    stream_key = f"FDQ:{cls_id}:{user_id}"    
    
    for file_info in files_info:
        file_id = file_info["file_id"]
        file_type_cd = file_info["file_type_cd"]
        file_path = file_info["file_path"]

        update_hash_and_notify_task(stream_key, "embedding", file_id, {"embedding_status": "start"})

        try:
            # 파일 존재 및 확장자 확인
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                update_hash_and_notify_task(stream_key, "embedding", file_id, {"embedding_status": "failed", "error": "파일을 찾을 수 없습니다."})                
                # 파일 업로드 실패 코드로 설정, dwld_fail_dt에 시간값 넣기
                await update_status_in_db("FU04", 'DWLD_FAIL_DT', file_id)
                continue

            _, ext = os.path.splitext(file_path)
            if not ext:
                print(f"No file extension found for: {file_path}")
                update_hash_and_notify_task(stream_key, "embedding", file_id, {"embedding_status": "failed", "error": "파일 확장자가 없습니다."})                
                # 임베딩 처리 불가 코드로 설정, 시간값 업데이트 없음
                await update_status_in_db("EP05", 'NO_UPDATE', file_id)
                continue

            # 파일 로드 및 태그 추가
            try:
                documents = await load_file(file_path, file_type_cd, file_id)
                await update_status_in_db("EP02", 'EMB_START_DT', file_id)
                print(f"{file_id} 파일 로드 성공!")
            except ValueError as ve:
                if "지원되지 않는 파일 형식" in str(ve):
                    # 파일 형식 오류는 임베딩 처리 불가로 기록
                    print(f"File format error for {file_id}: {str(ve)}")
                    update_hash_and_notify_task(stream_key, "embedding", file_id, 
                                    {"embedding_status": "failed", "error": str(ve)})
                    await update_status_in_db("EP05", 'NO_UPDATE', file_id)
                    continue
                else:
                    # 다른 ValueError는 다시 던져서 임베딩 실패로 처리
                    raise

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            docs = text_splitter.split_documents(documents)

            # 빈 문서 필터링
            docs = [doc for doc in docs if doc.page_content.strip()]

            # 메타데이터 추가 및 텍스트 전처리
            today = datetime.now().strftime("%Y-%m-%d")
            for doc in docs:
                doc.metadata["title"] = documents[0].metadata.get("title", "Unknown")
                doc.metadata["종류"] = documents[0].metadata.get("종류", "Unknown")
                doc.metadata["embedding_date"] = today
                doc.metadata["file_id"] = file_id
                # 텍스트에 메타데이터 정보 추가
                doc.page_content = preprocess_text_with_metadata(doc.page_content, doc.metadata)

            # 문서가 비어있는 경우 기본 문서 추가
            if not docs:
                default_doc = Document(
                    page_content=preprocess_text_with_metadata("", documents[0].metadata),
                    metadata={
                        "title": documents[0].metadata.get("title", "Unknown"),
                        "종류": documents[0].metadata.get("종류", "Unknown"),
                        "embedding_date": today,
                        "file_id": file_id
                    }
                )
                docs = [default_doc]

            # 임베딩 생성 및 저장
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                
                Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name=cls_id,
                    client=CHROMA_CLIENT # 서버방식으로 변경
                )

            except Exception as chroma_e:
                print(f"ChromaDB 오류: {chroma_e}")
                raise
            
            # 임베딩 성공 처리
            await update_status_in_db("EP03", 'EMB_COMP_DT', file_id)
            update_hash_and_notify_task(stream_key, "embedding", file_id, {"embedding_status": "completed"})
            print(f"{file_id} 파일 임베딩 성공!")

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 예외 메시지에 따라 파일 처리 실패 또는 임베딩 실패로 구분
            if "지원되지 않는 파일 형식" in str(e) or "No file extension" in str(e):
                # 파일 형식 관련 오류는 임베딩 처리 불가로 기록 (시간값 업데이트 없음)
                await update_status_in_db("EP05", 'NO_UPDATE', file_id)
                error_message = "파일 형식 오류 (임베딩 처리 불가): " + str(e)
            else:
                # 그 외 오류는 임베딩 실패로 기록
                await update_status_in_db("EP04", 'EMB_FAIL_DT', file_id)
                error_message = "임베딩 실패: " + str(e)
                
            update_hash_and_notify_task( stream_key, "embedding", file_id,
                              {"embedding_status": "failed", "error": error_message},
                              {"status": "failed", "error": error_message})            
            print(f"{file_id} 처리 실패: {error_message}")

    print("Processing completed successfully")

async def process_and_embed_new_files(cls_id: str, user_id: str):
    """
    새 파일 처리 및 임베딩을 수행하는 메인 함수
    """
    try:
        # 1. 새로운 파일 식별 (FU03 상태의 파일)
        new_files = await identify_new_files(cls_id)
        if not new_files:
            print("No new files to process")
            return
            
        print(f"Found {len(new_files)} files to process")
        
        # 2. 파일 정보 처리
        files_info = []
        for file in new_files:
            file_info = {
                "file_name": file['file_nm'],
                "file_id": file['file_id'],
                "file_type_cd": file['file_type_cd'],
                "file_path": file['file_path']
            }
            files_info.append(file_info)
            print(f"Processing file: {file_info}")
        
        # 3. 임베딩 진행
        await process_and_embed_files(cls_id, files_info, user_id)
        
        print("Processing completed successfully")
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

def update_hash_and_notify_task(stream_key: str, task_type:str, file_id: str, update_fields: dict, extra_fields: dict = None):
    try:
        status_info = {}
        status_info.update(update_fields)
        message = {
            "task_type": task_type,
            "file_id": file_id
        }
        if extra_fields:
            message.update(extra_fields)
        redis_client.xadd(stream_key, message, id='*')
    except Exception as e:
        print(f"Redis update error for {file_id}: {e}")

# 예제 실행
if __name__ == "__main__":
    json_data = {
        "cls_id": "2024-2-17156-01"
        ,"user_id": "45932"
    }

    asyncio.run(process_and_embed_new_files(json_data["cls_id"],json_data["user_id"]))
