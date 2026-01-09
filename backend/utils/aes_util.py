import base64
import os
from Crypto.Cipher import AES
from dotenv import load_dotenv

load_dotenv(dotenv_path="/app/tutor/.env")
SECRET_KEY = os.getenv("AES_SECRET_KEY")

# 보이지 않는 제어문자 제거
clean_key = "".join(ch for ch in SECRET_KEY if 32 <= ord(ch) <= 126)
clean_key = clean_key[:32].ljust(32, "0")


# AES 준비
BLOCK_SIZE = 16

def pad(data: bytes):
    """AES 블록 크기(16)에 맞게 패딩 추가"""
    padding = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([padding]) * padding

def encrypt(text: str) -> str:
    """AES ECB 모드 암호화"""
    cipher = AES.new(clean_key.encode(), AES.MODE_ECB)
    padded = pad(text.encode())
    encrypted = cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode()

def unpad(data: bytes):
    """AES 블록 크기(16)에 맞게 패딩 제거"""
    return data[:-data[-1]]

def decrypt(encrypted_text: str) -> str:
    """AES 복호화 (base64 패딩 오류 자동 보정)"""
    try:
        missing_padding = len(encrypted_text) % 4
        if missing_padding:
            encrypted_text += '=' * (4 - missing_padding)

        decoded = base64.b64decode(encrypted_text)
        cipher = AES.new(clean_key.encode(), AES.MODE_ECB)
        decrypted = cipher.decrypt(decoded)
        return unpad(decrypted).decode("utf-8")

    except Exception as e:
        print(f"[DECRYPT ERROR] {encrypted_text[:30]}... → {e}")
        raise e