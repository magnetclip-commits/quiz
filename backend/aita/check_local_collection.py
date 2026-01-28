import os
from langchain_chroma import Chroma
import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
# CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))
CHROMA_HOST = os.getenv("CHROMA_HOST", "hlta-chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embeddings = OpenAIEmbeddings()
cls_id = '2025-20-002033-1-01'
#cls_id = '2025-1-903186-01'
#client_path = os.path.join(CHROMA_CLIENT, cls_id)
vectorstore = Chroma(
    client=CHROMA_CLIENT,
    embedding_function=embeddings,
    collection_name=cls_id,
)
# vectorstore._collection은 chromadb.Collection 객체
col = vectorstore._collection

# 전체 문서 수
print("Total docs:", col.count())

# 일부만 보기
rows = col.get(limit=10, include=["documents", "metadatas"])
ids = rows.get("ids") or []
docs = rows.get("documents") or []
metas = rows.get("metadatas") or []

for i, _id in enumerate(ids, 1):
    meta = metas[i-1] if i-1 < len(metas) else {}
    preview = (docs[i-1] or "")[:200].replace("\n", " ")
    print(f"[{i}] id={_id} meta={meta} text≈{repr(preview)}")
