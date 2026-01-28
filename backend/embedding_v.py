'''
@modified: embedding_v_0411_HJ
2025.07.28 상태코드별 웹소켓 메세지 전송
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
from langchain.chains.summarize import load_summarize_chain # LangChain 관련 임포트
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
from langchain.prompts import PromptTemplate
import redis
from utils.milvus_util import construct_milvus_payload, send_milvus_ingest

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

def preprocess_text_with_category(text, file_type_str, title):
    """
    파일 내용 앞에 종류 정보를 추가하여 임베딩할 때 활용할 수 있도록 변환
    """
    category_text = f"이 문서는 교수자가 [{file_type_str}] 카테고리에 업로드한 내용입니다. 참고한 파일의 제목은 [{title}]입니다. \n\n"
    return category_text + text

def load_text_file(file_path, title=None, file_type_str=None):
    """
    텍스트 파일 로드 및 처리
    """
    loader = TextLoader(file_path)
    documents = loader.load()

    # 메타데이터 추가 + 종류 정보 포함한 텍스트 변환
    for doc in documents:
        doc.page_content = preprocess_text_with_category(doc.page_content, file_type_str, title)
        if title:
            doc.metadata["title"] = title
        if file_type_str:
            doc.metadata["종류"] = file_type_str
    return documents


    
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

async def update_status_in_db(status, dt_tag, file_id, cls_id=None, user_id=None):
    """
    데이터베이스에서 파일 상태 업데이트 및 웹소켓 알림 전송
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
        
        # 웹소켓 알림 전송
        if cls_id and user_id:
            stream_key = f"FDQ:{cls_id}:{user_id}"        
            redis_hash_key = f"{cls_id}:material_status"
            
            # 상태에 따라 적절한 메시지 생성
            status_message = ""
            task_type = "status_update"
            
            if status == "EP02":
                status_message = "튜터학습중"
                task_type = "embedding"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"embedding_status": "processing", "message": status_message})
            elif status == "EP03":
                status_message = "튜터학습완료"
                task_type = "embedding"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"embedding_status": "completed", "message": status_message})
            elif status == "EP04":
                status_message = "튜터학습실패"
                task_type = "embedding"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"embedding_status": "failed", "message": status_message})
            elif status == "SM02":
                status_message = "주차별학습중"
                task_type = "summary"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"summary_status": "processing", "message": status_message})
            elif status == "SM03":
                status_message = "주차별학습완료"
                task_type = "summary"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"summary_status": "completed", "message": status_message})
            elif status == "SM04":
                status_message = "주차별학습실패"
                task_type = "summary"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"summary_status": "failed", "message": status_message})
            elif status == "FU04":
                status_message = "업로드 실패"
                task_type = "upload"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"upload_status": "failed", "message": status_message})
            elif status == "EP05":
                status_message = "임베딩 처리 불가"
                task_type = "embedding"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"embedding_status": "unsupported", "message": status_message})
            elif status == "FD01":
                status_message = "삭제중"
                task_type = "delete"
                update_hash_and_notify_task(redis_hash_key, stream_key, task_type, file_id, {"delete_status": "processing", "message": status_message})
            
    except Exception as e:
        print(f"DB 업데이트 오류: {file_id} - {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            await conn.close()

# 주차 정보 추출 (week_num 추출)
def extract_week_info(file_path: str) -> int:
    """
    파일 경로에서 주차 정보 추출
    예: /uploaded_files/CLASS_ID/week_1/file.pdf -> 1
    """
    try:
        path_parts = file_path.split('/')
        for part in path_parts:
            if part.lower().startswith('week'):
                week_num = ''.join(filter(str.isdigit, part))
                return int(week_num) if week_num else 0
        return 0
    except Exception:
        return 0

def preprocess_text_with_metadata(text: str, metadata: dict) -> str:
    """
    텍스트에 메타데이터 정보를 추가
    """
    week = metadata.get('week', '알 수 없음')
    file_type = metadata.get('종류', '알 수 없음')
    title = metadata.get('title', '알 수 없음')
    
    prefix = f"이 문서는 {week}주차에 교수자가 [{file_type}] 카테고리에 업로드한 내용입니다. "
    prefix += f"참고한 파일의 제목은 [{title}]입니다.\n\n"
    
    return prefix + text

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
            stt_file_path as file_path,
            upload_status
        FROM cls_file_mst
        WHERE cls_id = $1
        AND upload_status IN ('FS03')  -- 파일 업로드 완료인 파일 찾기
        AND file_ext = 'mp4' -- 영상 파일만 가져오기 
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

async def update_summaries_for_newly_embedded_files(cls_id: str, file_path: str, docs: list, user_id: str = None):
    try:
        # 주차 정보를 file_path에서 추출하는 대신 docs의 메타데이터에서 가져옵니다
        # 모든 docs는 같은 파일에서 왔으므로 첫 번째 doc의 week 값을 사용합니다
        if not docs or len(docs) == 0:
            print("No documents to process")
            return
        
        # 파일 ID 추출
        file_id = docs[0].metadata.get("file_id", "")
        if not file_id:
            print("No file_id found in metadata")
            return
            
        # 요약 시작 상태 업데이트
        await update_status_in_db("SM02", 'SUMRY_START_DT', file_id, cls_id=cls_id, user_id=user_id)
        print(f"Summary process started for file_id: {file_id}")

        week = docs[0].metadata.get("week", 0)
        if week <= 0:
            # 메타데이터에서 유효한 주차 정보를 찾지 못한 경우 DB에서 조회
            try:
                file_id = docs[0].metadata.get("file_id", "")
                if file_id:
                    conn = await asyncpg.connect(**DATABASE_CONFIG)
                    try:
                        query = """
                            SELECT week_num
                            FROM cls_file_mst
                            WHERE file_id = $1
                        """
                        week = await conn.fetchval(query, file_id)
                        week = int(week) if week is not None else 0
                        print(f"Retrieved week {week} from database for file_id: {file_id}")
                    finally:
                        await conn.close()
            except Exception as db_err:
                print(f"Error fetching week from database: {db_err}")
                # 요약 실패 상태 업데이트
                await update_status_in_db("SM04", 'SUMRY_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
                print(f"Summary process failed for file_id: {file_id}")
                return
                            
        if week <= 0:
            print("No valid week information found")
            # 요약 실패 상태 업데이트
            await update_status_in_db("SM04", 'SUMRY_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
            print(f"Summary process failed for file_id: {file_id} - No valid week information")
            return
            
        print(f"Creating summary for week: {week}")
            
        # ChromaDB에서 같은 주차의 모든 문서 검색
        #chroma_db_path = os.path.join("./db/chromadb", cls_id)
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            #persist_directory=chroma_db_path,
            client=CHROMA_CLIENT, # 서버방식으로 변경
            embedding_function=embeddings, 
            collection_name=cls_id
            )
        
        # 현재 주차의 모든 문서 검색
        results = db.get(
            where={"week": week}
        )
        
        if not results or not results['documents']:
            print(f"No documents found in ChromaDB for week {week}")
            documents = docs  # 현재 처리 중인 문서만 사용
        else:
            # ChromaDB의 문서와 현재 문서 결합
            documents = []
            for i, content in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            documents.extend(docs)  # 현재 처리 중인 문서 추가
            
        print(f"Total {len(documents)} documents found for week {week}")
        
        # 요약을 위한 프롬프트 템플릿 정의 (간결하게 수정)
        map_prompt_template = """대학 강의 수업 자료입니다. 핵심 개념, 중요 용어, 주요 내용을 간결하게 요약해주세요.

{text}

요약:"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        combine_prompt_template = """대학 강의 주차별 수업 자료 요약입니다. 아래 요약들을 통합하여 핵심 내용을 3-4개 섹션으로 구성해 주세요.

{text}

종합 요약:"""
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        # 요약 생성
        try:
            llm = ChatOpenAI(temperature=0)
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=False
            )
            chain_result = chain.invoke(documents)
            summary = chain_result.get('output_text', '') if isinstance(chain_result, dict) else str(chain_result)
            
            # 타이틀 생성 (한국어)
            title_prompt = PromptTemplate(
                template="다음 대학 강의 요약의 핵심 주제를 반영한 간결한 제목을 한 줄로 작성해주세요:\n\n{text}\n\n제목:",
                input_variables=["text"]
            )
            title_result = llm.invoke(title_prompt.format(text=summary[:500]))
            title = title_result.content.strip() if hasattr(title_result, 'content') else str(title_result)
        except Exception as llm_err:
            print(f"Error generating summary with LLM: {llm_err}")
            # 요약 실패 상태 업데이트
            await update_status_in_db("SM04", 'SUMRY_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
            print(f"Summary process failed for file_id: {file_id}")
            return
            
        # JSON 형식으로 데이터 구성
        content_data = json.dumps({
            "title": title,
            "content": summary
        }, ensure_ascii=False)
        
        # 현재 시간 가져오기
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # DB에 요약 저장 (UPDATE 또는 INSERT)
        query = """
            WITH upsert AS (
                UPDATE cls_weekly_learn 
                SET content_data = $3,
                    upd_dt = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
                WHERE cls_id = $1 
                AND week_num = $2
                RETURNING *
            )
            INSERT INTO cls_weekly_learn (cls_id, week_num, content_data, ins_dt, upd_dt)
            SELECT $1, $2, $3, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul', CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
            WHERE NOT EXISTS (SELECT * FROM upsert);
        """
        
        conn = None
        try:
            conn = await asyncpg.connect(**DATABASE_CONFIG)
            await conn.execute(
                query, 
                cls_id,    # $1
                week,      # $2
                content_data  # $3
            )
            print(f"Week {week} summary updated successfully")
            
            # 요약 완료 상태 업데이트
            await update_status_in_db("SM03", 'SUMRY_COMP_DT', file_id, cls_id=cls_id, user_id=user_id)
            print(f"Summary process completed for file_id: {file_id}")
        except Exception as db_error:
            print(f"Error updating summary in database: {db_error}")
            # 요약 실패 상태 업데이트
            await update_status_in_db("SM04", 'SUMRY_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
            print(f"Summary process failed for file_id: {file_id}")
            raise
        finally:
            if conn:
                await conn.close()
        
        return f"Updated summary for week {week}"
        
    except Exception as e:
        print(f"Error updating summaries: {e}")
        import traceback
        print(traceback.format_exc())

        # 파일 ID가 있는 경우 실패 상태 업데이트
        if docs and len(docs) > 0:
            file_id = docs[0].metadata.get("file_id", "")
            if file_id:
                await update_status_in_db("SM04", 'SUMRY_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
                print(f"Summary process failed for file_id: {file_id}")
                
        raise

# async def process_and_embed_files(cls_id: str, files_info: list, notify_filestatus_toclients):
async def process_and_embed_files(cls_id: str, files_info: list, user_id: str):
    """
    파일 처리 및 임베딩 수행
    """
    base_path = f"/app/tutor/download_files/{cls_id}"
    #chroma_db_path = os.path.join("./db/chromadb", cls_id)

    # 디렉토리 권한 확인 및 설정
    ensure_directory_permissions(base_path)
        
    stream_key = f"FDQ:{cls_id}:{user_id}"    
    redis_hash_key = f"{cls_id}:material_status"
    
    for file_info in files_info:
        file_id = file_info["file_id"]
        file_type_cd = file_info["file_type_cd"]
        file_path = file_info["file_path"]

        update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "start"})

        try:
            # 파일 존재 및 확장자 확인
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "failed", "error": "파일을 찾을 수 없습니다."})                
                # 파일 업로드 실패 코드로 설정, dwld_fail_dt에 시간값 넣기
                await update_status_in_db("FU04", 'DWLD_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
                redis_client.hdel(redis_hash_key, file_id)
                continue

            _, ext = os.path.splitext(file_path)
            if not ext:
                print(f"No file extension found for: {file_path}")
                update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "failed", "error": "파일 확장자가 없습니다."})                
                # 임베딩 처리 불가 코드로 설정, 시간값 업데이트 없음
                await update_status_in_db("EP05", 'NO_UPDATE', file_id, cls_id=cls_id, user_id=user_id)
                redis_client.hdel(redis_hash_key, file_id)
                continue

            # 파일 로드 및 태그 추가
            try:
                documents = await load_file(file_path, file_type_cd, file_id)
                await update_status_in_db("EP02", 'EMB_START_DT', file_id, cls_id=cls_id, user_id=user_id)
                print(f"{file_id} 파일 로드 성공!")
            except ValueError as ve:
                if "지원되지 않는 파일 형식" in str(ve):
                    # 파일 형식 오류는 임베딩 처리 불가로 기록
                    print(f"File format error for {file_id}: {str(ve)}")
                    update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, 
                                    {"embedding_status": "failed", "error": str(ve)})
                    await update_status_in_db("EP05", 'NO_UPDATE', file_id, cls_id=cls_id, user_id=user_id)
                    redis_client.hdel(redis_hash_key, file_id)
                    continue
                else:
                    raise

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # 메타데이터에 주차 정보와 임베딩 날짜 추가
            today = datetime.now().strftime("%Y-%m-%d")
            for doc in docs:
                doc.metadata["title"] = documents[0].metadata.get("title", "Unknown")
                doc.metadata["종류"] = documents[0].metadata.get("종류", "Unknown")
                doc.metadata["week"] = extract_week_info(file_path)
                doc.metadata["embedding_date"] = today  # 임베딩 날짜 추가
                doc.metadata["file_id"] = file_id  # 파일 ID 추가
                # 텍스트에 주차 정보 추가
                doc.page_content = preprocess_text_with_metadata(doc.page_content, doc.metadata)

            # 임베딩 생성 및 저장
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

                # Retry logic for dimension mismatch
                max_retries = 1
                for attempt in range(max_retries + 1):
                    try:
                        Chroma.from_documents(
                            documents=docs,
                            embedding=embeddings,
                            client=CHROMA_CLIENT, # 서버방식으로 변경
                            collection_name=cls_id
                        )
                        break # Success
                    except Exception as e:
                         # Catch dimension mismatch and retry
                        if "dimension" in str(e) and "1536" in str(e) and "3072" in str(e):
                            if attempt < max_retries:
                                print(f"Warning: Dimension mismatch detected for {cls_id} (Video). Deleting and recreating collection... (Attempt {attempt+1})")
                                CHROMA_CLIENT.delete_collection(cls_id)
                                continue
                            else:
                                print(f"Error: Dimension mismatch persists after retry for {cls_id}.")
                                raise e
                        else:
                            raise e

            except Exception as chroma_e:
                print(f"ChromaDB 오류: {chroma_e}")
                raise
            
            # 임베딩 성공 처리 및 요약문 작성
            if doc.metadata["week"] > 0:
                # 먼저 임베딩 완료 상태 업데이트
                await update_status_in_db("EP03", 'EMB_COMP_DT', file_id, cls_id=cls_id, user_id=user_id)
                
                # 요약문 작성 
                print(f"{file_id} 파일 임베딩 완료, 요약 작성 시작...")
                await update_summaries_for_newly_embedded_files(cls_id, file_path, docs, user_id)
                
                update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "completed"})
                print(f"{file_id} 파일 임베딩 및 요약 생성 성공!")
            else:
                # 주차 정보가 없는 경우에는 요약 작성 없이 DB 업데이트
                update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id, {"embedding_status": "completed"})
                print(f"{file_id} 파일 임베딩 성공! (주차 정보 없음)")

            # --- Real-time Milvus Sync Trigger ---
            try:
                print(f"[{file_id}] Real-time Milvus Sync Trigger...")
                conn = await asyncpg.connect(**DATABASE_CONFIG)
                try:
                    query = """
                        SELECT 
                            f.file_id, f.cls_id, f.file_type_cd, f.file_nm, f.file_ext, 
                            f.file_path, f.stt_file_path, f.file_size, f.week_num, c.user_id
                        FROM cls_file_mst f
                        JOIN cls_mst c ON f.cls_id = c.cls_id
                        WHERE f.file_id = $1
                    """
                    row = await conn.fetchrow(query, file_id)
                    if row:
                        payload = construct_milvus_payload(dict(row))
                        success, response_text = send_milvus_ingest(payload)
                        status = 'Y' if success else 'E'
                        await conn.execute(
                            "UPDATE cls_file_mst SET milvus_yn = $1, milvus_upd_dt = NOW() WHERE file_id = $2", 
                            status, file_id
                        )
                        print(f" -> Milvus Sync Result: {status}")
                finally:
                    await conn.close()
            except Exception as sync_e:
                print(f"Milvus Sync Trigger Error for {file_id}: {sync_e}")
            
            redis_client.hdel(redis_hash_key, file_id)

        except Exception as e:
            print(f"Error processing file {file_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 예외 메시지에 따라 파일 처리 실패 또는 임베딩 실패로 구분
            if "지원되지 않는 파일 형식" in str(e) or "No file extension" in str(e):
                # 파일 형식 관련 오류는 임베딩 처리 불가로 기록 (시간값 업데이트 없음)
                await update_status_in_db("EP05", 'NO_UPDATE', file_id, cls_id=cls_id, user_id=user_id)
                error_message = "파일 형식 오류 (임베딩 처리 불가): " + str(e)
            else:
                # 그 외 오류는 임베딩 실패로 기록
                await update_status_in_db("EP04", 'EMB_FAIL_DT', file_id, cls_id=cls_id, user_id=user_id)
                error_message = "임베딩 실패: " + str(e)
                
            update_hash_and_notify_task(redis_hash_key, stream_key, "embedding", file_id,
                              {"embedding_status": "failed", "error": error_message},
                              {"status": "failed", "error": error_message})
            
            redis_client.hdel(redis_hash_key, file_id)
            print(f"{file_id} 처리 실패: {error_message}")

# async def process_new_videos_and_update_summaries(cls_id: str, notify_filestatus_toclients):
async def process_new_videos_and_update_summaries(cls_id: str, user_id: str):
    """
    새 파일 처리 및 주차별 요약 업데이트를 수행하는 메인 함수
    """
    try:
        # 1. 새로운 파일 식별 (FU03, EP04 상태의 파일)
        new_files = await identify_new_files(cls_id)
        if not new_files:
            print("No new files to process")
            return
            
        print(f"Found {len(new_files)} files to process")
        
        # 2. 파일 정보 처리 및 임베딩 진행
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

# 예제 실행
if __name__ == "__main__":
    json_data = {
        "cls_id": "2024-2-17156-01"
        ,"user_id": "45932"
    }

    # notify_filestatus_toclients를 동기 함수로 처리
    # asyncio.run(process_new_videos_and_update_summaries(json_data["cls_id"], lambda x: print(x)))
    asyncio.run(process_new_videos_and_update_summaries(json_data["cls_id"],json_data["user_id"]))


# 정리 : raw_file 이 경로에 들어와 있고, FS03(업로드 완료 된 상태)일때 임베딩 -> 새롭게 임베딩이 된 파일이 속한 주차에 대해서는 요약문 다시 만듦