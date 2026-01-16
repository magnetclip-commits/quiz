import chromadb
import os

# 환경 변수 또는 직접 주소 입력 (현재 8002 포트 사용 중)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))

try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    collections = client.list_collections()
    total_vectors = 0

    print(f"{'Collection Name':<40} | {'Count':<10}")
    print("-" * 55)

    for col in collections:
        count = col.count()
        total_vectors += count
        print(f"{col.name:<40} | {count:<10}")

    print("-" * 55)
    print(f"Total Collections: {len(collections)}")
    print(f"Total Vectors: {total_vectors}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
