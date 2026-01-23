import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException, status
from typing import Optional
import logging

from config import JWT_SECRET_KEY as SECRET_KEY, JWT_ALGORITHM as ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1시간
REFRESH_TOKEN_EXPIRE_DAYS = 7    # 7일

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # PyJWT 2.x+ 호환성: 문자열 반환
    return encoded_jwt if isinstance(encoded_jwt, str) else encoded_jwt.decode('utf-8')

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # PyJWT 2.x+ 호환성: 문자열 반환
    return encoded_jwt if isinstance(encoded_jwt, str) else encoded_jwt.decode('utf-8')

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logging.warning("JWT token expired")
        return None
    except jwt.PyJWTError as e:
        logging.error(f"JWT decode error: {e}")
        return None

async def get_current_user_optional(request: Request) -> Optional[str]:
    """
    Hybrid Auth: Authorization 헤더가 있으면 검증하여 user_id(sub) 반환, 
    없으면 None 반환하여 기존 Body 기반 인증이 작동하도록 함.
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ")[1]
    payload = decode_token(token)
    
    if not payload or payload.get("type") != "access":
        return None
        
    user_id: str = payload.get("sub") or payload.get("user_id") # 과도기적 지원
    return user_id
