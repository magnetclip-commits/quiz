import requests
import json
import os

# Configuration
BASE_URL = "http://211.188.58.37:8085"
CLS_ID = "TEST_CLASS_001"
USER_ID = "verified_user"
FILE_PATH = "test_upload.txt"

# Create dummy test file
with open(FILE_PATH, "w") as f:
    f.write("This is a test file for on-demand embedding verification.")

def test_upload_and_embed():
    print(f"Testing against {BASE_URL}...")

    # 1. Upload File
    upload_url = f"{BASE_URL}/file/uploadfiles"
    
    metadata = {
        "cls_id": CLS_ID,
        "file_nm": FILE_PATH,
        "file_ext": "txt",
        "file_format": "text/plain",
        "file_size": os.path.getsize(FILE_PATH),
        "upload_status": "FU01"
    }

    files = {
        'files': (FILE_PATH, open(FILE_PATH, 'rb'), 'text/plain')
    }
    data = {
        'metadata': json.dumps(metadata)
    }

    print(f"\n[1] Uploading file to {upload_url}...")
    try:
        resp = requests.post(upload_url, files=files, data=data)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code != 200:
            print("Upload failed.")
            return

        result = resp.json()
        saved_files = result.get("files", [])
        if not saved_files:
            print("No files returned in response.")
            return
            
        file_id = saved_files[0]['file_id']
        print(f"File Uploaded Successfully. File ID: {file_id}")
        
    except Exception as e:
        print(f"Upload Exception: {e}")
        return

    # 2. Trigger Embedding
    embed_url = f"{BASE_URL}/file/embedding"
    payload = {
        "cls_id": CLS_ID,
        "user_id": USER_ID
    }
    
    print(f"\n[2] Triggering embedding at {embed_url}...")
    try:
        resp = requests.post(embed_url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code == 200:
            print("Embedding triggered successfully.")
        else:
            print("Embedding trigger failed.")
            
    except Exception as e:
        print(f"Embedding Exception: {e}")

if __name__ == "__main__":
    test_upload_and_embed()
    # Cleanup
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)
