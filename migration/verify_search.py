import chromadb

# 1. 서버 접속
client = chromadb.HttpClient(host="hlta-chroma", port=8000)
collection_name = "여기에_컬렉션_이름_입력" # (예: lecture_notes)
collection = client.get_collection(collection_name)

# 2. 변경된 내용 검색 (질의)
query_text = "이번에 새로 추가된 강의 내용 키워드"  # <--- 여기를 수정하세요
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

# 3. 결과 출력
print("=== 검색 결과 (Top 3) ===")
for i, doc in enumerate(results['documents'][0]):
    print(f"[{i+1}] {doc}")
    print("-" * 30)
