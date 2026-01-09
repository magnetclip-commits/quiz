import os
import chromadb
from tqdm import tqdm

# === 설정 ===
SOURCE_DIR = "/app/chromadb_source"
TARGET_HOST = "hlta-chroma"
TARGET_PORT = 8000
BATCH_SIZE = 500  # [중요] 5000 -> 500으로 축소 (Payload 에러 방지)
# ============

def migrate():
    print(f"🚀 Safe Migration Start (Batch: {BATCH_SIZE})")
    
    # 1. 소스 DB 로드
    try:
        src_client = chromadb.PersistentClient(path=SOURCE_DIR)
        collections = src_client.list_collections()
        print(f"🔍 Found {len(collections)} collections.")
    except Exception as e:
        print(f"❌ Failed to load source DB: {e}")
        return

    # 2. 타겟 연결
    try:
        target_client = chromadb.HttpClient(host=TARGET_HOST, port=TARGET_PORT)
        target_client.heartbeat()
    except Exception as e:
        print(f"❌ Target Connection Failed: {e}")
        return

    success_cnt = 0
    error_cnt = 0

    # 3. 마이그레이션 루프
    for col in tqdm(collections, desc="Processing"):
        try:
            data = col.get(include=["embeddings", "documents", "metadatas"])
            total_docs = len(data['ids'])
            
            if total_docs == 0:
                continue

            # 타겟 컬렉션 생성
            dest_col = target_client.get_or_create_collection(
                name=col.name,
                metadata=col.metadata
            )

            # === 배치 전송 ===
            for i in range(0, total_docs, BATCH_SIZE):
                end = i + BATCH_SIZE
                
                # 데이터가 None인 경우 처리 (안전장치)
                b_metadatas = data['metadatas'][i:end] if data['metadatas'] else None
                b_documents = data['documents'][i:end] if data['documents'] else None
                
                dest_col.add(
                    ids=data['ids'][i:end],
                    embeddings=data['embeddings'][i:end],
                    metadatas=b_metadatas,
                    documents=b_documents
                )
            # ================

            success_cnt += 1
            
        except Exception as inner_e:
            print(f"\n⚠️ FAILED on '{col.name}': {inner_e}")
            error_cnt += 1

    print("\n" + "="*40)
    print(f"🎉 Result")
    print(f"✅ Success: {success_cnt}")
    print(f"❌ Failed: {error_cnt}")
    print("="*40)

if __name__ == "__main__":
    migrate()
