import os
import sys
from pymilvus import connections, Collection, utility

# Standalone Milvus Verification Script
# This script only requires 'pymilvus' to be installed.

def verify_file_in_milvus(file_id: str):
    # Configuration - matches hlta/backend/retriever/retriever.env
    MILVUS_HOST = "kaai-milvus-standalone"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "hallym_dev9"

    print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        
        if not utility.has_collection(COLLECTION_NAME):
            print(f"[!] Collection '{COLLECTION_NAME}' not found.")
            return

        collection = Collection(COLLECTION_NAME)
        collection.load()

        print(f"Querying collection '{COLLECTION_NAME}' for file_id (field or metadata): {file_id}...")
        
        # 1. Try direct field match
        res = collection.query(
            expr=f'file_id == "{file_id}"',
            output_fields=["file_id", "title", "text", "cls_id", "metadata"],
            limit=5
        )

        # 2. If not found, try searching inside metadata JSON for file_id or title
        if not res:
            try:
                # Search by title in metadata (JobManager uses 'title' or 'filename' in metadata usually)
                # We'll try to find any record that has the file_id in its metadata
                res = collection.query(
                    expr=f'metadata["file_id"] == "{file_id}"',
                    output_fields=["file_id", "title", "text", "cls_id", "metadata"],
                    limit=5
                )
            except:
                pass
        
        if not res:
            try:
                # Based on the job input, the title was "_1주차 대면수업_파일.mp4"
                res = collection.query(
                    expr='metadata["title"] like "%주차%"',
                    output_fields=["file_id", "title", "text", "cls_id", "metadata"],
                    limit=5
                )
                print(f"Found {len(res)} general matches for '주차' in title.")
            except:
                pass

        if not res:
            print(f"[-] No data found in Milvus for file_id: {file_id}")
            print("\n--- Diagnostic Information ---")
            collections = utility.list_collections()
            print(f"Available Collections: {collections}")
            
            for coll_name in collections:
                c = Collection(coll_name)
                print(f" - Collection '{coll_name}': {c.num_entities} entities")
                
                # Sample some data to see what's actually inside
                if c.num_entities > 0:
                    print(f"\n--- Deep Dive into '{coll_name}' ---")
                    
                    # 1. Search for our IDs in ANY field (using term search on metadata if possible)
                    # Note: Milvus query doesn't support 'contains' well on strings, but we can check if it's in metadata JSON
                    # Instead, we'll just pull the latest 10 entities to see what was last added.
                    
                    # 2. Pull the most recent entries (if there's a timestamp or just top 10)
                    latest = c.query(expr="", output_fields=["file_id", "cls_id", "metadata"], limit=10)
                    print(f"   [Latest 10 entries]")
                    for j, r in enumerate(latest):
                        meta_shorthand = str(r.get('metadata'))[:100] + "..." if r.get('metadata') else "None"
                        print(f"   {j+1}. file_id: {r.get('file_id')}, cls_id: {r.get('cls_id')}")
                        print(f"      Metadata: {meta_shorthand}")

                    # 3. Check for our CLS_ID specifically in a query
                    test_cls_id_short = file_id.split("-")[0] # e.g. "2025"
                    print(f"\n   Searching for any record containing '{test_cls_id_short}' in cls_id...")
                    matches = c.query(expr=f'cls_id like "%{test_cls_id_short}%"', output_fields=["file_id", "cls_id"], limit=5)
                    print(f"   Matches: {matches}")
                    
            print("\nNote: Ingestion is asynchronous. If the 'Latest' list doesn't show your data, it might still be in the JobAPI queue.")
            print("Note: If the JobAPI translates IDs to UUIDs, your file_id might be inside the 'metadata' JSON field.")
        else:
            print(f"[+] Found {len(res)} chunks for file_id: {file_id}")
            for i, row in enumerate(res):
                print(f"\n--- Chunk {i+1} ---")
                print(f"Cls ID: {row.get('cls_id')}")
                print(f"Title: {row.get('title')}")
                print(f"Text Preview: {row.get('text')[:200]}...")

    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_milvus_data.py <file_id>")
        sys.exit(1)
        
    target_file_id = sys.argv[1]
    verify_file_in_milvus(target_file_id)
