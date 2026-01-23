from fastapi import APIRouter, Depends
from config import DATABASE_CONFIG, DATABASE2_CONFIG, FRONTEND_URL
from fastapi import Request, HTTPException
import asyncpg
import oracledb
from contextlib import contextmanager
import uuid
from datetime import datetime, timedelta, timezone
from fastapi.responses import RedirectResponse
import logging
import bcrypt
import re

from utils.aes_util import decrypt, encrypt
from utils.security import create_access_token, get_current_user_optional


router = APIRouter()
BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/=]+$')

@router.get("/check")
async def check_api():
    return {"message": "OK"}

@router.get("/verify")
async def verify_token(user_id: str = Depends(get_current_user_optional)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return {"user_id": user_id, "status": "valid"}

@router.post("/decrypt-password")
async def decrypt_password(request: Request):
    """
    암호화된 비밀번호를 평문으로 복호화
    """
    body = await request.json()
    encrypted_pw = body.get("encrypted_password")
    
    if not encrypted_pw:
        raise HTTPException(status_code=400, detail="encrypted_password is required")
    
    try:
        decrypted_pw = decrypt(encrypted_pw)
        return {
            "status": "success",
            "encrypted_password": encrypted_pw,
            "decrypted_password": decrypted_pw
        }
    except Exception as e:
        logging.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Decryption failed: {str(e)}")

# 로그인 처리 API
@router.post("/login")
async def login(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    user_pw = body.get("user_pw")

    query = """
    SELECT *
    FROM user_mst
    WHERE user_id = $1;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1, $2)에 값 매핑
            row = await conn.fetchrow(query, user_id)
            if not row:
                raise HTTPException(status_code=401, detail="User not found")
            
            stored_pw = row["user_pw"]
            decrypted_pw = None

            # 기존 DB에 PW bcrypt 로 되어있어 AES로 새로 저장
            if stored_pw.startswith("$2a$") or stored_pw.startswith("$2b$"):
                print(f"[INFO] bcrypt detected for {user_id}")
                if bcrypt.checkpw(user_pw.encode(), stored_pw.encode()):
                    encrypted_pw = encrypt(user_pw)
                    await conn.execute(
                        "UPDATE user_mst SET user_pw = $1 WHERE user_id = $2",
                        encrypted_pw,
                        user_id
                    )
                    print(f"[INFO] bcrypt → AES migration done for {user_id}")
                    decrypted_pw = user_pw
                else:
                    raise HTTPException(status_code=401, detail="Invalid password")

            # AES 암호화된 문자열인 경우
            elif BASE64_PATTERN.match(stored_pw):
                try:
                    decrypted_pw = decrypt(stored_pw)
                except Exception as e:
                    print(f"[DECRYPT ERROR] {stored_pw[:20]}... → {e}")
                    raise HTTPException(status_code=500, detail=f"Decrypt failed: {str(e)}")

                if decrypted_pw != user_pw:
                    raise HTTPException(status_code=401, detail="Invalid password")

            # 둘 다 아닌 경우 (평문)
            else:
                print(f"[WARN] plain password detected for {user_id}")
                if stored_pw == user_pw:
                    encrypted_pw = encrypt(user_pw)
                    await conn.execute(
                        "UPDATE user_mst SET user_pw = $1 WHERE user_id = $2",
                        encrypted_pw,
                        user_id
                    )
                    print(f"[INFO] plain → AES migration done for {user_id}")
                else:
                    raise HTTPException(status_code=401, detail="Invalid password")
            if decrypted_pw == user_pw:
                user_id = row["user_id"]
                access_token = create_access_token(data={"user_id": user_id})
                return {
                        "user_id": user_id,
                        "user_div": row["user_div"],
                        "user_nm": row["user_nm"],
                        "access_token": access_token,
                        "token_type": "bearer"
                }
            else:
                raise HTTPException(status_code=401, detail="login failure")
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# 한림대 통합시스템 로그인 처리 API
@contextmanager
def get_db_connection():
    connection = oracledb.connect(**DATABASE2_CONFIG)
    try:
        yield connection
    finally:
        if connection:
            connection.close()

@router.post("/login_h")
async def login_h(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    user_pw = body.get("user_pw")

    if not user_id or not user_pw:
        raise HTTPException(status_code=400, detail="Missing user_id or user_pw")

    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            ref_cursor = connection.cursor()
            
            try:
                sql = """
                BEGIN
                    :1 := ILBAN.NF_USER_LOGIN_INFO_MORE(:2, :3);
                END;
                """
                
                cursor.execute(sql, [ref_cursor, user_id, user_pw])
                row = ref_cursor.fetchone()
                
                if row:
                    columns = ['JAEJ_IDNO', 'JAEJ_NAME', 'JAEJ_DEPT', 'JAEJ_DEPT_NAME', 
                            'JAEJ_SOSOK', 'JAEJ_SOSOK_NAME', 'JAEJ_PRIV_WRITING', 
                            'JAEJ_PWD', 'JAEJ_DIV', 'ERR_MSG']
                    
                    response_data = dict(zip(columns, row))
                    return response_data
                
                return {"ERR_MSG": "0"}
                
            finally:
                if cursor:
                    cursor.close()
                if ref_cursor:
                    ref_cursor.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

   

@router.post("/update-password")
async def update_password(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    user_pw = body.get("user_pw")

    if not all([user_id, user_pw]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            check_query = "SELECT user_id FROM user_mst WHERE user_id = $1"
            row = await conn.fetchrow(check_query, user_id)
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            
            encrypted_pw = encrypt(user_pw)

            update_pw_query = """
                UPDATE user_mst
                SET user_pw = $1
                WHERE user_id = $2
            """
            row = await conn.execute(update_pw_query, encrypted_pw, user_id)

            if row.startswith("UPDATE"):
                return {
                    "user_id": user_id,
                    "message": "success"
                }
            else:
                raise HTTPException(status_code=500, detail="Database error")
        except HTTPException:
            raise
        finally:
            await conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
