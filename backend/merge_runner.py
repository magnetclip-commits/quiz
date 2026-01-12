import os
import chromadb
from tqdm import tqdm
from datetime import datetime

# === 설정 ===
# 마운트된 수신 데이터 경로 (hltutor의 DB)
SOURCE_DIR = "/tmp/tutorchroma" 
# 현재 운영 중인 hlta의 ChromaDB (여기에 합침)
TARGET_HOST = "hlta-chroma"
TARGET_PORT = 8000
BATCH_SIZE = 500  # 안정적인 전송을 위한 배치 사이즈
# ============

def merge_db():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕒 Merge Timestamp for this batch: {current_time}")
    print(f"🚀 Merge Start: [File] {SOURCE_DIR} -> [Server] {TARGET_HOST}")

    # 1. Source (File) 로드
    if not os.path.exists(os.path.join(SOURCE_DIR, "chroma.sqlite3")):
        print(f"❌ Source DB not found in {SOURCE_DIR}. Run rsync first.")
        return

    try:
        src_client = chromadb.PersistentClient(path=SOURCE_DIR)
        collections = src_client.list_collections()
        print(f"🔍 Source Collections Found: {len(collections)}")
    except Exception as e:
        print(f"❌ Failed to load source DB: {e}")
        return

    # 2. Target (Server) 연결
    try:
        target_client = chromadb.HttpClient(host=TARGET_HOST, port=TARGET_PORT)
        target_client.heartbeat()
        print("✅ Target Server Connected")
    except Exception as e:
        print(f"❌ Target Connection Failed: {e}")
        return

    success_cnt = 0

    # 3. 데이터 병합 (Upsert)
    for col in tqdm(collections, desc="Merging Collections"):
        try:
            # Source 데이터 가져오기
            data = col.get(include=["embeddings", "documents", "metadatas"])
            total_docs = len(data['ids'])
            
            if total_docs == 0:
                continue

            # Target 컬렉션 준비 (없으면 생성)
            dest_col = target_client.get_or_create_collection(
                name=col.name,
                metadata=col.metadata
            )

            # 배치 단위 Upsert (Insert가 아니라 Upsert 사용!)
            for i in range(0, total_docs, BATCH_SIZE):
                end = i + BATCH_SIZE
                
                b_ids = data['ids'][i:end]
                b_embeddings = data['embeddings'][i:end]
                b_documents = data['documents'][i:end] if data['documents'] else None
                # [수정됨] 메타데이터에 날짜 정보 추가 로직 ---------------------------
                # 원본 메타데이터 슬라이싱 (없으면 None 리스트로 대체하여 인덱스 맞춤)
                raw_metas = data['metadatas'][i:end] if data['metadatas'] else [None] * len(b_ids)
                
                b_metadatas = []
                for meta in raw_metas:
                    # 기존 메타데이터가 있으면 복사, 없으면 빈 딕셔너리 생성
                    new_meta = meta.copy() if meta else {}
                    # 날짜 정보 강제 주입
                    new_meta['last_updated'] = current_time
                    b_metadatas.append(new_meta)
                # ------------------------------------------------------------------
                
                # upsert: ID가 같으면 덮어쓰고, 없으면 추가함 (Union 효과)
                dest_col.upsert(
                    ids=b_ids,
                    embeddings=b_embeddings,
                    metadatas=b_metadatas,
                    documents=b_documents
                )
            
            success_cnt += 1

        except Exception as inner_e:
            print(f"⚠️ Error merging '{col.name}': {inner_e}")

    print("="*40)
    print(f"🎉 Merge Completed. Collections processed: {success_cnt}")
    print("="*40)

if __name__ == "__main__":
    merge_db()
