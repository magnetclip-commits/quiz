from fastapi import APIRouter
from config import DATABASE_CONFIG
from fastapi import Request, HTTPException
import asyncpg
from typing import List, Dict, Tuple
import re
from collections import Counter
# from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 KoNLPy 사용
import hashlib
import time
import traceback
import logging
from utils.aes_util import encrypt

router = APIRouter()
logger = logging.getLogger(__name__)


# 수강목록 API
@router.post("/classlist")
async def classlist(request: Request):
    body = await request.json()
    cls_yr = body.get("cls_yr")
    cls_smt = body.get("cls_smt")
    user_id = body.get("user_id")

    logger.info(f"Classlist request with cls_yr={cls_yr}, cls_smt={cls_smt}, user_id={user_id}")
    logger.debug(f"DATABASE_CONFIG: {DATABASE_CONFIG}")

    # cls_smt 값에 따라 검색할 학기 목록 설정
    smt_values = []
    if cls_smt == "1":
        smt_values = ["1", "10"]
    elif cls_smt == "2":
        smt_values = ["2", "20"]
    else:
        smt_values = [cls_smt]

   # 일반 학기 쿼리와 공통 과목(0000/00)
    query = """
        SELECT E.USER_ID, E.CLS_ID, M.CLS_NM, UM.USER_NM, M.CLS_YR, M.CLS_SMT, M.CLS_NM_EN, M.RAG_YN, M.UPLOAD_TYPE
        FROM CLS_ENR E
        INNER JOIN CLS_MST M ON E.CLS_ID = M.CLS_ID
        LEFT OUTER JOIN USER_MST UM ON M.USER_ID = UM.USER_ID --교수님이름
        WHERE (M.CLS_YR = $1 AND M.CLS_SMT = ANY($2)  AND E.USER_ID = $3)
        OR (M.cls_id = '0000-00-000000-0-00' AND E.USER_ID = $3)
        ORDER BY cls_id
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("DB connection successful")
        try:
            # 매개변수 전달 시 자리표시자($1, $2, $3)에 값 매핑
            rows = await conn.fetch(query, cls_yr, smt_values, user_id)
            await conn.close() # inserted by mj

            if rows:
                # 결과를 리스트로 변환하여 반환
                return [
                    {"cls_id": row["cls_id"], "cls_nm": row["cls_nm"], "user_nm": row["user_nm"], "cls_nm_en": row["cls_nm_en"], "rag_yn": row["rag_yn"], "upload_type": row["upload_type"]}
                    for row in rows
                ]
            else:
                logger.warning(f"No classes found for user {user_id} in year {cls_yr} semester {cls_smt}") # mjo
                raise HTTPException(status_code=401, detail="No classes found")
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Database error: {repr(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        # logging.error("Exception in /classlist endpoint:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database error: {repr(e)}")
        # raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# 수강 학생 조회 API
@router.post("/cls/std/list")
async def class_student_list(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")

    query = """       
        select um.user_id , um.user_nm , um.dpt_nm 
        FROM user_mst um
        WHERE um.user_id IN (
            SELECT ce.user_id 
            FROM cls_enr ce 
            WHERE ce.cls_id = $1
            AND ce.user_div = 'S'
        );
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 매개변수 전달 시 자리표시자($1)에 값 매핑
            rows = await conn.fetch(query, cls_id)

            if rows:
                # 결과를 리스트로 변환하여 반환
                return [
                    {"user_id": row["user_id"], 
                     "user_nm": row["user_nm"], 
                     "dpt_nm": row["dpt_nm"]
                    }
                    for row in rows
                ]
            else:
                return []
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    

# 수강 학생 검색 API
@router.post("/search/std")
async def search_student(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    user_id = body.get("user_id")
    search_id = body.get("search_id")
    
    # LIKE 패턴에서 자리표시자 올바르게 사용
    query = """
    SELECT um.user_id, um.user_nm, um.dpt_nm 
    FROM user_mst um 
    WHERE univ_cd = $1 
    AND user_div = 'S' 
    AND (um.user_id LIKE $2 OR um.user_nm LIKE $2)
    AND um.user_id NOT IN (
        SELECT ce.user_id 
        FROM cls_enr ce 
        WHERE ce.cls_id = $3
    );
    """
    
    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # univ_cd 조회
            data_row = await conn.fetchrow("SELECT univ_cd FROM user_mst WHERE user_id = $1;", user_id)
            if not data_row:
                raise HTTPException(status_code=404, detail="User not found")
            
            univ_cd = data_row["univ_cd"]
            
            # search_id에 LIKE 패턴 적용 (% 기호 추가)
            search_pattern = f"%{search_id}%"
            
            # 여러 행을 가져오기 위해 fetch() 사용 (fetchrow() 대신)
            rows = await conn.fetch(query, univ_cd, search_pattern, cls_id)
            
            if not rows:
                return []
            
            # 모든 결과를 리스트로 반환
            return [
                {
                    "user_id": row["user_id"], 
                    "user_nm": row["user_nm"], 
                    "dpt_nm": row["dpt_nm"]
                } 
                for row in rows
            ]
            
        finally:
            await conn.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# SmartLead 계정 정보 등록/갱신 API (Upsert)
@router.post("/register")
async def register_user(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    user_pw = body.get("user_pw")
    user_nm = body.get("user_nm", "사용자") 
    
    # 추가 필드 (프론트엔드에서 전달받음)
    user_div = body.get("user_div", "S")   # 기본값: 학생(S)
    univ_cd = body.get("univ_cd", "001")   # 기본값: 001
    dpt_nm = body.get("dpt_nm", "")        # 학과명 (옵션)
    
    # 강의 목록 (옵션)
    classes = body.get("classes", [])

    if not all([user_id, user_pw]):
        raise HTTPException(status_code=400, detail="Missing user_id or user_pw")

    try:
        # 비밀번호 암호화
        encrypted_pw = encrypt(user_pw)

        # 1. User Upsert 쿼리
        user_query = """
            INSERT INTO user_mst (user_id, user_pw, user_nm, user_div, univ_cd, dpt_nm, ins_dt, upd_dt)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                user_pw = EXCLUDED.user_pw,
                upd_dt = NOW(),
                user_nm = COALESCE(EXCLUDED.user_nm, user_mst.user_nm),
                user_div = COALESCE(EXCLUDED.user_div, user_mst.user_div),
                univ_cd = COALESCE(EXCLUDED.univ_cd, user_mst.univ_cd),
                dpt_nm = COALESCE(EXCLUDED.dpt_nm, user_mst.dpt_nm)
        """
        
        # 2. Class Upsert 쿼리
        cls_query = """
            INSERT INTO cls_mst (cls_id, course_id, cls_nm, cls_sec, user_id, cls_yr, cls_smt, cls_grd, rag_yn, ins_dt, upd_dt)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'N', NOW(), NOW())
            ON CONFLICT (cls_id)
            DO UPDATE SET
                cls_nm = EXCLUDED.cls_nm,
                user_id = EXCLUDED.user_id,
                upd_dt = NOW(),
                rag_yn = user_mst.rag_yn -- 기존 RAG 여부 유지 (또는 필요시 갱신)
        """
        # 주의: 위 쿼리에서 user_mst.rag_yn -> cls_mst.rag_yn 오타 수정 필요.
        # DO UPDATE SET rag_yn = cls_mst.rag_yn (기존 값 유지)
        # 만약 신규 등록이면 'N'으로 들어감.
        
        cls_query_corrected = """
            INSERT INTO cls_mst (cls_id, course_id, cls_nm, cls_sec, user_id, cls_yr, cls_smt, cls_grd, rag_yn, ins_dt, upd_dt)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'N', NOW(), NOW())
            ON CONFLICT (cls_id)
            DO UPDATE SET
                cls_nm = EXCLUDED.cls_nm,
                user_id = EXCLUDED.user_id,
                upd_dt = NOW()
                -- rag_yn은 업데이트 하지 않음 (기존 상태 유지)
        """

        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 트랜잭션 시작 (다중 쿼리 안전성 보장)
            async with conn.transaction():
                # User 저장
                await conn.execute(user_query, user_id, encrypted_pw, user_nm, user_div, univ_cd, dpt_nm)
                
                # Classes 저장
                for cls in classes:
                    c_id = cls.get("cls_id")
                    c_course = cls.get("course_id", c_id.split('-')[2] if c_id and '-' in c_id else "unknown")
                    c_nm = cls.get("cls_nm", "Unknown Class")
                    c_sec = cls.get("cls_sec", "1")
                    c_yr = cls.get("year", "2025")
                    c_smt = cls.get("semester", "20")
                    c_grd = cls.get("cls_grd", "1")
                    
                    if c_id:
                        await conn.execute(cls_query_corrected, c_id, c_course, c_nm, c_sec, user_id, c_yr, c_smt, c_grd)

            logger.info(f"User {user_id} and {len(classes)} classes registered/updated successfully.")
            return {"status": "success", "message": f"User {user_id} and {len(classes)} classes processed."}
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Register User Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
