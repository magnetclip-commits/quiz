from fastapi import APIRouter
import os
from config import DATABASE_CONFIG
from fastapi import Request, HTTPException, Depends, Query
import asyncpg
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
# from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 KoNLPy 사용
import hashlib
import time
import traceback
import logging
from utils.aes_util import encrypt
from utils.security import get_current_user_optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Profile API
# ============================================================================

class ProfileResponse(BaseModel):
    user_id: str
    profile_type: str
    cls_id: Optional[str] = None
    profile_body: str
    use_yn: str
    created_dt: Optional[datetime] = None
    updated_dt: Optional[datetime] = None


class ProfileHistoryResponse(BaseModel):
    hist_id: int
    user_id: str
    profile_type: str
    cls_id: Optional[str] = None
    profile_body: str
    use_yn: str
    saved_dt: datetime


class UseYnUpdateRequest(BaseModel):
    use_yn: str  # 'Y' or 'N'


# ============================================================================
# Profile APIs
# ============================================================================

# 1. GET - 현재 교수 프롬프트 값 조회
@router.get("/api/profile/current/{user_id}", response_model=Optional[ProfileResponse])
async def get_current_profile(
    user_id: str,
    profile_type: str = Query("USR", description="Profile type (USR: 교수, CLS: 과목)"),
    cls_id: Optional[str] = Query(None, description="Class ID (profile_type=CLS인 경우 필수)"),
    jwt_user_id: str = Depends(get_current_user_optional)
):
    """
    현재 설정되어 있는 교수/과목 프롬프트 값 조회
    
    - **user_id**: 사용자 ID (교수 ID)
    - **profile_type**: USR(교수 프롬프트) 또는 CLS(과목 프롬프트)
    - **cls_id**: 과목 프롬프트 조회 시 필수
    """
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        query = """
            SELECT * 
            FROM aita_profile_mst apm 
            WHERE user_id = $1
            AND profile_type = $2
        """
        
        params = [user_id, profile_type]
        
        # CLS 타입인 경우 cls_id 추가 필터링
        if profile_type == "CLS" and cls_id:
            query += " AND cls_id = $3"
            params.append(cls_id)
        
        row = await conn.fetchrow(query, *params)
        await conn.close()
        
        if not row:
            return None
        
        return dict(row)
    
    except Exception as e:
        logger.error(f"Error fetching profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터 조회 실패: {str(e)}")


# 2. GET - 과거 교수 프롬프트 이력 조회
@router.get("/api/profile/history/{user_id}", response_model=List[ProfileHistoryResponse])
async def get_profile_history(
    user_id: str,
    profile_type: str = Query("USR", description="Profile type (USR: 교수, CLS: 과목)"),
    cls_id: Optional[str] = Query(None, description="Class ID (profile_type=CLS인 경우 선택)"),
    limit: int = Query(10, description="최대 조회 개수", ge=1, le=100),
    jwt_user_id: str = Depends(get_current_user_optional)
):
    """
    과거 교수/과목 프롬프트 변경 이력 조회 (최신순)
    
    - **user_id**: 사용자 ID (교수 ID)
    - **profile_type**: USR(교수 프롬프트) 또는 CLS(과목 프롬프트)
    - **cls_id**: 과목 프롬프트 조회 시 선택사항
    - **limit**: 최대 조회 개수 (1-100, 기본값: 10)
    """
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        query = """
            SELECT * 
            FROM aita_profile_hist aph  
            WHERE user_id = $1
            AND profile_type = $2
        """
        
        params = [user_id, profile_type]
        param_idx = 3
        
        # CLS 타입인 경우 cls_id 추가 필터링
        if profile_type == "CLS" and cls_id:
            query += f" AND cls_id = ${param_idx}"
            params.append(cls_id)
            param_idx += 1
        
        query += f" ORDER BY hist_id DESC LIMIT ${param_idx}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        await conn.close()
        
        return [dict(row) for row in rows]
    
    except Exception as e:
        logger.error(f"Error fetching profile history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이력 조회 실패: {str(e)}")


# 3. PUT - 교수 프롬프트 사용여부 변경
@router.put("/api/profile/professor/{user_id}/toggle-use")
async def update_professor_profile_use(
    user_id: str,
    request: UseYnUpdateRequest,
    jwt_user_id: str = Depends(get_current_user_optional)
):
    """
    교수 프롬프트 사용여부 변경
    
    - **user_id**: 교수 ID
    - **use_yn**: 'Y' (사용) 또는 'N' (미사용)
    """
    # use_yn 유효성 검증
    if request.use_yn not in ['Y', 'N']:
        raise HTTPException(status_code=400, detail="use_yn은 'Y' 또는 'N'이어야 합니다")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        query = """
            UPDATE aita_profile_mst
            SET use_yn = $1, prof_mod_dt = NOW()
            WHERE user_id = $2
            AND profile_type = 'USR'
            RETURNING *
        """
        
        result = await conn.fetchrow(query, request.use_yn, user_id)
        await conn.close()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="해당 사용자의 교수 프롬프트가 존재하지 않습니다"
            )
        
        logger.info(f"Professor profile use_yn updated for user {user_id} to {request.use_yn}")
        return {
            "message": "교수 프롬프트 사용여부가 변경되었습니다",
            "data": dict(result)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating professor profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"업데이트 실패: {str(e)}")


# 4. PUT - 과목 프롬프트 사용여부 변경
@router.put("/api/profile/class/{user_id}/{cls_id}/toggle-use")
async def update_class_profile_use(
    user_id: str,
    cls_id: str,
    request: UseYnUpdateRequest,
    jwt_user_id: str = Depends(get_current_user_optional)
):
    """
    과목 프롬프트 사용여부 변경
    
    - **user_id**: 교수 ID
    - **cls_id**: 강의 ID
    - **use_yn**: 'Y' (사용) 또는 'N' (미사용)
    """
    # use_yn 유효성 검증
    if request.use_yn not in ['Y', 'N']:
        raise HTTPException(status_code=400, detail="use_yn은 'Y' 또는 'N'이어야 합니다")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        query = """
            UPDATE aita_profile_mst
            SET use_yn = $1, prof_mod_dt = NOW()
            WHERE user_id = $2
            AND cls_id = $3
            AND profile_type = 'CLS'
            RETURNING *
        """
        
        result = await conn.fetchrow(query, request.use_yn, user_id, cls_id)
        await conn.close()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="해당 강의의 과목 프롬프트가 존재하지 않습니다"
            )
        
        logger.info(f"Class profile use_yn updated for user {user_id}, class {cls_id} to {request.use_yn}")
        return {
            "message": "과목 프롬프트 사용여부가 변경되었습니다",
            "data": dict(result)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating class profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"업데이트 실패: {str(e)}")


# ============================================================================
# Existing APIs
# ============================================================================
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
                        # [NEW] 강좌 등록 후 프로필 초기화 SP 호출
                        try:
                            await conn.execute("CALL sp_aita_profile_init($1, $2);", user_id, c_id)
                            logger.info(f"Initialized profile for {user_id} in {c_id}")
                        except Exception as sp_e:
                            logger.error(f"SP execution failed for {c_id}: {sp_e}")
                            # SP 실패가 전체 가입 실패로 이어지지 않게 하거나, 
                            # 혹은 트랜잭션을 깨야 한다면 raise를 그대로 둡니다.
                            # 여기서는 주요 데이터(User/Class) 저장이 우선이므로 로깅 후 진행하도록 설정할 수 있지만, 
                            # 트랜잭션 내부이므로 raise 시 자동 롤백됩니다.
                            raise sp_e 

            logger.info(f"User {user_id} and {len(classes)} classes registered/updated successfully.")
            return {"status": "success", "message": f"User {user_id} and {len(classes)} classes processed."}
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Register User Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# 유저 프로필 초기화 API
@router.post("/profile/init")
async def init_user_profile(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    cls_id = body.get("cls_id")

    if not user_id or not cls_id:
        raise HTTPException(status_code=400, detail="user_id와 cls_id는 필수 항목입니다.")

    logger.info(f"Profile initialization request for user_id={user_id}, cls_id={cls_id}")

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 프로시저 호출: sp_aita_profile_init(user_id, cls_id)
            # CALL 구문을 사용하여 프로시저를 실행합니다.
            await conn.execute("CALL sp_aita_profile_init($1, $2);", user_id, cls_id)
            
            logger.info(f"Successfully initialized profile for user_id={user_id}")
            return {
                "status": "success",
                "message": "사용자 프로필이 성공적으로 초기화되었습니다. (hlta)"
            }
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Profile Init Error (sp_aita_profile_init): {repr(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Database error while calling sp_aita_profile_init: {repr(e)}")


# YAML 프로필 경로 설정
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aita", "yaml", "USR")

# 프로필 YAML 조회 API
@router.get("/profile/config/{user_id}")
async def get_profile_config(user_id: str):
    # 보안 검증: user_id가 영숫자/대시/밑줄인지 확인 (Directory Traversal 방지)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    file_path = os.path.join(PROFILE_DIR, f"{user_id}.yaml")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Profile YAML not found")
    
    try:
        # aita/yaml 가 gitignore 되어 있을 수 있으므로 직접 읽기
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"user_id": user_id, "yaml_content": content}
    except Exception as e:
        logger.error(f"Error reading profile YAML for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading profile file")

# 프로필 YAML 저장 API
@router.post("/profile/config")
async def save_profile_config(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    yaml_content = body.get("yaml_content")

    if not user_id or yaml_content is None:
        raise HTTPException(status_code=400, detail="Missing user_id or yaml_content")

    # 보안 검증
    if not re.match(r"^[a-zA-Z0-9_\-]+$", user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    file_path = os.path.join(PROFILE_DIR, f"{user_id}.yaml")
    
    # 디렉토리가 없으면 생성 (안전장치)
    if not os.path.exists(PROFILE_DIR):
        os.makedirs(PROFILE_DIR, exist_ok=True)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"Profile YAML for {user_id} updated successfully.")
        return {"status": "success", "message": "Profile updated successfully"}
    except Exception as e:
        logger.error(f"Error saving profile YAML for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error saving profile file")

    
