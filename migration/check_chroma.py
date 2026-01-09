import chromadb
import os

# 1. 실제 ChromaDB가 저장된 경로를 지정하세요. (보통 hltutor 내의 persist directory)
# 예: persist_path = "./chroma_db" 또는 sync_chroma_union.sh에서 지정한 경로
persist_path = "/opt/hlta/tutorchroma"

# 2. 클라이언트 연결
if os.path.exists(persist_path):
    client = chromadb.PersistentClient(path=persist_path)
    print(f"✅ DB 경로 확인됨: {persist_path}")
else:
    print(f"❌ 경로를 찾을 수 없습니다: {persist_path}")
    exit()

# 3. 컬렉션 리스트 확인
collections = client.list_collections()
print(f"📂 총 컬렉션 수: {len(collections)}")

for col in collections:
    print(f"\n--- Collection: {col.name} ---")
    
    # 4. 데이터 개수 확인 (이전보다 증가했거나 예상되는 수치인지 확인)
    count = col.count()
    print(f"📊 데이터 개수(Count): {count}")
    
    # 5. 최신 데이터 샘플 확인 (수정한 로직이 반영된 데이터가 있는지)
    # peek()는 상위 몇 개의 데이터를 가져옵니다.
    if count > 0:
        peek_data = col.peek(limit=3)
        print(f"🔍 샘플 데이터(Metadatas): {peek_data['metadatas']}")
