'''
@modified: 
2025.08.03 절대경로로 변경
2025.08.08 chroma, openaiembeddings 라이브러리 변경 
2025.08.14 chromadb 서버방식으로 변경
'''
import os
import asyncio
import asyncpg
from langchain_chroma import Chroma
import chromadb
#from langchain.vectorstores import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import DATABASE_CONFIG, OPENAI_API_KEY

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

async def update_status_in_db(status, dt_tag, file_id):
    """데이터베이스 상태 업데이트 함수"""
    query = f"""
        UPDATE CLS_FILE_MST
        SET UPLOAD_STATUS = $1,
        {dt_tag} = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
        WHERE FILE_ID = $2
    """
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            await conn.execute(query, status, file_id)
        finally:
            await conn.close()
        print(f"DB 업데이트: {file_id} - {status}")
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise


async def delete_embedded_materials(cls_id: str, files_info: list):
    """여러 파일의 임베딩 데이터 삭제를 처리하는 함수"""
    #chroma_db_path = os.path.join("/app/tutor/db/chromadb", cls_id) # 절대경로로 변경
    #chroma_db_path = os.path.join("./db/chromadb", cls_id)
    results = []

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            #persist_directory=chroma_db_path,
            client=CHROMA_CLIENT, # 서버방식으로 변경
            embedding_function=embeddings,
            collection_name=cls_id
        )

        all_docs = vector_store.get()
        ids_to_delete = []
        
        # 더 많은 확장자 지원
        supported_extensions = [".pdf", ".txt", ".ppt", ".pptx", ".mp4", ".docx", ".doc"]

        for file_info in files_info:
            file_id = file_info["file_id"]
            week_num = file_info.get("week_num", "")  # week_num이 없을 수 있음
            
            try:
                await update_status_in_db("FD01", "FILE_DEL_REQ_DT", file_id)

                # 파일 ID를 기반으로 한 더 유연한 매칭
                file_ids = [
                    all_docs['ids'][i]
                    for i, metadata in enumerate(all_docs['metadatas'])
                    if (file_id in metadata.get('source', '') or  # 소스 경로에서 매칭
                        metadata.get('file_id', '') == file_id)   # 파일 ID 직접 매칭
                ]
                
                if file_ids:
                    ids_to_delete.extend(file_ids)
                    await update_status_in_db("FD03", "EMB_DEL_COMP_DT", file_id)
                    print(f"Found and marked embeddings for file: {file_id} as deleted")
                    results.append({"file_id": file_id, "success": True})
                else:
                    print(f"No embeddings found for file: {file_id}")
                    
                    # DB에서 emb_fail_dt 컬럼 확인
                    try:
                        conn = await asyncpg.connect(**DATABASE_CONFIG)
                        query = """
                            SELECT EMB_FAIL_DT FROM CLS_FILE_MST
                            WHERE CLS_ID = $1 AND FILE_ID = $2
                        """
                        row = await conn.fetchrow(query, cls_id, file_id)
                        await conn.close()

                        if row and row['emb_fail_dt']:
                            await update_status_in_db("FD03", "EMB_DEL_COMP_DT", file_id)
                            print(f"File {file_id} was previously marked as embedding failure. Status updated.")
                            results.append({"file_id": file_id, "success": True})
                        else:
                            results.append({"file_id": file_id, "success": False})

                    except Exception as e:
                        print(f"Database error while checking EMB_FAIL_DT: {str(e)}")
                        results.append({"file_id": file_id, "success": False, "error": str(e)})

            except Exception as e:
                print(f"Error processing file {file_id}: {str(e)}")
                results.append({"file_id": file_id, "success": False, "error": str(e)})

        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} documents...")
            vector_store.delete(ids=ids_to_delete)
            #vector_store.persist()
            print(f"Successfully deleted all found embeddings")

        return results

    except Exception as e:
        print(f"ChromaDB 초기화 실패: {str(e)}")
        for file_info in files_info:
            results.append({
                "file_id": file_info["file_id"],
                "success": False,
                "error": f"ChromaDB initialization failed: {str(e)}"
            })

    return results


# 실행 예시
if __name__ == "__main__":
    json_data = {
        "cls_id": "2025-1-209106-01",
        "files_info": [
            {
                "file_id": "2025-1-209106-01-8afbe9",
                "week_num" : "week_2"
            }
        ]
    }
    
    results = asyncio.run(delete_embedded_materials(json_data["cls_id"], json_data["files_info"]))
    print("삭제 결과:", results)