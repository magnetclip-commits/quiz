import os
import asyncio
import asyncpg
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import DATABASE_CONFIG, OPENAI_API_KEY


async def update_status_in_db(status, dt_tag, file_id):
    """데이터베이스 상태 업데이트 함수"""
    query = f"""
        UPDATE FILE_MST
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


async def delete_embedded_files(cls_id: str, files_info: list):
    """여러 파일의 임베딩 데이터 삭제를 처리하는 함수"""
    chroma_db_path = os.path.join("./db/chromadb", cls_id)
    results = []

    try:
        # Chroma 클라이언트 초기화 (한 번만 수행)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # OPENAI_API_KEY는 자동으로 로드됨
        vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
            collection_name=cls_id
        )

        # 모든 문서 조회
        all_docs = vector_store.get()

        # 삭제할 문서 ID들을 저장할 리스트
        ids_to_delete = []
        
        # 지원하는 파일 확장자 목록
        supported_extensions = [".pdf", ".txt", ".ppt", ".pptx"]

        # 각 파일별 처리
        for file_info in files_info:
            file_id = file_info["file_id"]
            expected_sources = [f"./uploaded_files/{cls_id}/{file_id}{ext}" for ext in supported_extensions]

            try:
                # 삭제 시작 상태 업데이트
                await update_status_in_db("FD01", "FILE_DEL_REQ_DT", file_id)

                # 현재 파일에 해당하는 문서 ID 찾기
                file_ids = [
                    all_docs['ids'][i]
                    for i, metadata in enumerate(all_docs['metadatas'])
                    if metadata.get('source', '') in expected_sources
                ]
                
                if file_ids:
                    # 삭제할 ID 목록에 추가
                    ids_to_delete.extend(file_ids)
                    await update_status_in_db("FD03", "EMB_DEL_COMP_DT", file_id)
                    print(f"Found and marked embeddings for file: {file_id} as deleted")
                    results.append({"file_id": file_id, "success": True})
                else:
                    print(f"No embeddings found for file: {file_id}")
                    print(f"Looked for source paths: {expected_sources}")

                    # DB에서 emb_fail_dt 컬럼 확인
                    try:
                        conn = await asyncpg.connect(**DATABASE_CONFIG)
                        query = f"""
                            SELECT EMB_FAIL_DT FROM FILE_MST
                            WHERE CLS_ID = $1 AND FILE_ID = $2
                        """
                        row = await conn.fetchrow(query, cls_id, file_id)
                        await conn.close()

                        if row and row['emb_fail_dt']:  # 임베딩 실패 이력이 있는 경우
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

        # 모든 파일 처리 후 한 번에 삭제 수행
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
            vector_store.persist()
            print(f"Successfully deleted all found embeddings")

        return results

    except Exception as e:
        print(f"ChromaDB 초기화 실패: {str(e)}")
        # ChromaDB 초기화 실패 시 모든 파일에 대해 실패 처리
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
        "cls_id": "2025-1-511644-01",
        "files_info": [
            {
                "file_name": "2025-1-511644-01-29587e.pdf",
                "file_id": "2025-1-511644-01-29587e"
            }
        ]
    }
    
    results = asyncio.run(delete_embedded_files(json_data["cls_id"], json_data["files_info"]))
    print("삭제 결과:", results)