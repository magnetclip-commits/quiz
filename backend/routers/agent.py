from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import traceback
from config import DATABASE_CONFIG
import asyncpg
import json
import hashlib
import pytz
from datetime import datetime
from typing import Optional
from aita.session_log import process_chat_session
from aita.quiz_create_v2 import quizmain
from aita.scoring_team import RunScoringAgentTeam
#from aita.imageProcessor import mainReceiptPreprocessor
from ocr_testpaper.ocr_gpt import ocr_main
import logging
import random
import string

router = APIRouter()


#사용자 질문 API
@router.post("/session/log")
async def session_log(request: Request):
    try:
        body = await request.json()
        
        # 필수 필드 검증
        required_fields = {
            "user_id": "사용자 ID",
            "cls_id": "강의 ID",
            "session_id": "세션 ID",
            "question": "유저 질문"
        }
        
        # 모든 필수 필드 존재 여부와 값 검증
        for field, field_name in required_fields.items():
            if field not in body or not body[field]:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_name}은(는) 필수 입력값입니다"
                )
        
        # 유저 요청 처리
        result = await process_chat_session(body["user_id"], body["cls_id"], body["session_id"], body["question"])
        
        return result
        
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"유저 요청 처리 중 오류가 발생했습니다: {str(e)}"
        )
    

#문항 생성 API
@router.post("/quiz/create")
async def quiz_create(request: Request):
    try:
        body = await request.json()
        
        # 필수 필드 검증
        required_fields = {
            "user_id": "사용자 ID",
            "cls_id": "강의 ID",
            "session_id": "세션 ID",
            "chat_seq":"채팅 시퀀스 번호",
            "question": "유저 질문"
        }
        
        # 공통 필수 필드 존재 검사
        for field, field_name in required_fields.items():
            if field not in body:
                raise HTTPException(status_code=400, detail=f"{field_name}은(는) 필수 입력값입니다")

        if body["chat_seq"] is None or not isinstance(body["chat_seq"], int) or body["chat_seq"] < 0:
            raise HTTPException(status_code=400, detail="채팅 시퀀스 번호는 0 이상의 정수여야 합니다")

        
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            course_row = await conn.fetchrow("SELECT course_id, cls_nm FROM cls_mst WHERE cls_id = $1", body["cls_id"])
            if not course_row:
                raise HTTPException(status_code=404, detail="cls_id not found")
            course_id = course_row["course_id"]
            subject_name = course_row["cls_nm"]
        finally:
            await conn.close()
            
        exam_config = {
            "subject_name": subject_name,
            "course_id": course_id,
            "custom_request": body["question"]
        }

        # 퀴즈 생성 실행
        result = await quizmain(body["user_id"], body["cls_id"], body["session_id"], body["chat_seq"], exam_config)
        
        return result["exam_data"]
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}"
        )

#유저 요청 처리 API
@router.post("/user/request/quiz")
async def user_request_quiz(request: Request):
    logger = logging.getLogger(__name__)
    try:
        body = await request.json()

        # 1) 공통 필수값 검증
        required_fields = {
            "user_id": "사용자 ID",
            "cls_id": "강의 ID",
            # "session_id": "세션 ID",
            "question": "유저 질문",
        }
        for f, name in required_fields.items():
            if f not in body or not body[f]:
                logger.warning(f"Missing required field: {f}")
                raise HTTPException(status_code=400, detail=f"{name}은(는) 필수 입력값입니다")

        user_id    = body["user_id"]
        cls_id     = body["cls_id"]
        session_id = None
        question   = body["question"]

        logger.info(f"[/user/request/quiz] Processing: user={user_id}, cls={cls_id}, q='{question}'")

        # 2) session/log 처리
        logger.info("Calling process_chat_session...")
        log_result = await process_chat_session(user_id, cls_id, session_id, question)
        logger.info(f"log_result: {log_result}")

        # 2-1) 최소 검증 (형/필드)
        if not isinstance(log_result, dict):
            logger.error(f"log_result is not dict: {type(log_result)} = {log_result}")
            raise HTTPException(status_code=502, detail="세션 로그 응답이 유효한 JSON 객체가 아닙니다")
        # status = log_result.get("status")
        logger.info(f"log_result keys: {list(log_result.keys())}")

        # do it when the status is success; mjo 
        status = log_result.get("status")
        if status != "success":
            error_msg = log_result.get("error", "Unknown error")
            logger.error(f"process_chat_session FAILED: status={status}, error={error_msg}")
            raise HTTPException(status_code=502, detail=f"세션 처리 실패: {error_msg}")

        for k in ["session_id", "seq", "status"]:
            if k not in log_result:
                logger.error(f"Missing required key '{k}' in log_result: {log_result}")
                raise HTTPException(status_code=502, detail=f"세션 로그 응답에 '{k}' 필드가 없습니다")
        if not isinstance(log_result["seq"], int) or log_result["seq"] < 0:
            logger.error(f"Invalid seq: {log_result['seq']} (type: {type(log_result['seq'])})")
            raise HTTPException(status_code=502, detail="세션 로그 응답의 seq가 유효한 정수가 아닙니다")
        if str(log_result["status"]).lower() != "success":
            raise HTTPException(status_code=502, detail=f"세션 로그 처리 실패(status={log_result['status']})")

        chat_seq = log_result["seq"]  # <- 주신 응답의 seq를 그대로 사용
        session_id = log_result["session_id"]
        logger.info(f"Session processed: session_id={session_id}, seq={chat_seq}")

        # 3) 과목 메타 조회
        logger.info(f"Querying cls_mst for cls_id={cls_id}")
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            row = await conn.fetchrow(
                "SELECT course_id, cls_nm FROM cls_mst WHERE cls_id = $1",
                cls_id
            )
            if not row:
                logger.warning(f"cls_id not found: {cls_id}")
                raise HTTPException(status_code=404, detail="cls_id not found")
            course_id    = row["course_id"]
            subject_name = row["cls_nm"]
            logger.info(f"Course found: {course_id} - {subject_name}")
        finally:
            await conn.close()

        # 4) 퀴즈 생성 파라미터 구성 
        exam_config = {
            "subject_name":   subject_name,
            "course_id":      course_id,
            "custom_request": question,
            "file_title":     log_result.get("file_title"),
        }
        logger.info(f"Quiz config: {exam_config}")
        
        # 5) 퀴즈 생성 (멱등키: (session_id, chat_seq))
        logger.info("Calling quizmain...")
        quiz_result = await quizmain(user_id, cls_id, session_id, chat_seq, exam_config)
        logger.info(f"quiz_result: {quiz_result}")
        if not isinstance(quiz_result, dict) or "exam_data" not in quiz_result:
            raise HTTPException(status_code=502, detail="퀴즈 생성 결과가 비어 있거나 형식이 올바르지 않습니다")

        # 6) 최종 응답 (로그 메타 + 퀴즈 데이터 동시 반환)
        logger.info("Quiz generation SUCCESS")
        return {
            "status": "success",
            "session_log": log_result,            # 받은 로그를 그대로 돌려줌
            "exam_data":  quiz_result["exam_data"]
        }

    except HTTPException:
        logger.exception("HTTPException in /user/request/quiz")
        raise
    except Exception as e:
        # 1. (필수) 서버 로그에 Full Traceback을 남겨서 디버깅에 사용
        logger.exception(f"Critical error in /user/request/quiz: {e}")
        # logging.error(f"[/agent/user/request/quiz] Critical Error: {e}", exc_info=True)
        
        # 2. 응답(Response)에 에러 유형과 메시지를 포함
        error_type = type(e).__name__
        # str(e)가 비어있을 경우를 대비해 repr(e)를 사용하거나, str(e) 결과를 그대로 사용합니다.
        error_message = str(e)
        
        raise HTTPException(
            status_code=500,
            detail=f"세션 처리 및 퀴즈 생성 중 오류가 발생했습니다: {str(e)}"
        )
    
#기생성 문항 조회 API
@router.post("/item/data/llm")
async def item_data_llm(request: Request):
    try:
        body = await request.json()
        item_ids = body.get("item_ids")

        # 유효성 검사
        if not item_ids or not isinstance(item_ids, list):
            raise HTTPException(status_code=400, detail="item_ids는 리스트 형식이어야 합니다.")

        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # aita_chatbot_session 테이블에서 item_id가 일치하는 객체들을 추출
            # jsonb_array_elements()로 questions 배열을 펼치고, 필요한 항목만 필터링
            sql = """
                SELECT
                    q->>'item_id'         AS item_id,
                    q->>'item_content'    AS item_content,
                    q->'item_choices'     AS item_choices,
                    q->>'item_answer'     AS item_answer,
                    q->>'item_explain'    AS item_explain,
                    q->>'item_type_cd'    AS item_type_cd,
                    q->>'item_diff_cd'    AS item_diff_cd,
                    q->>'ins_dt'          AS ins_dt,
                    q->>'upd_dt'          AS upd_dt
                FROM aita_chatbot_session,
                     jsonb_array_elements(chat_file_json->'quiz_data'->'questions') AS q
                WHERE q->>'item_id' = ANY($1::text[]);
            """

            rows = await conn.fetch(sql, item_ids)

            # asyncpg Record → dict 변환
            results = [dict(r) for r in rows]

            return results

        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"기생성 문항 조회 중 오류: {str(e)}")
    
#기생성 문항 수정 API
@router.post("/item/edit/llm")
async def item_edit_llm(request: Request):
    try:
        body = await request.json()

        # 필수값
        item_id = body.get("item_id")
        if not item_id:
            raise HTTPException(status_code=400, detail="item_id는 필수입니다.")

        # 수정 허용 키
        updatable_keys = [
            "item_content",
            "item_choices",
            "item_answer",
            "item_explain",
            "item_diff_cd",
        ]
        new_fields = {k: body[k] for k in updatable_keys if k in body}

        payload_jsonb = json.dumps(new_fields, ensure_ascii=False)

        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            async with conn.transaction():
                # aita_chatbot_session 테이블 업데이트
                sql = """
                    UPDATE aita_chatbot_session
                    SET chat_file_json = jsonb_set(
                        chat_file_json,
                        '{quiz_data,questions}',
                        (
                            SELECT jsonb_agg(
                                CASE
                                    WHEN q->>'item_id' = $1
                                        THEN q || $2::jsonb     -- 전달된 키만 부분 덮어쓰기
                                    ELSE q
                                END
                            )
                            FROM jsonb_array_elements(chat_file_json->'quiz_data'->'questions') AS q
                        )
                    )
                    WHERE jsonb_path_exists(
                        chat_file_json,
                        '$.quiz_data.questions[*] ? (@.item_id == $id)',
                        jsonb_build_object('id', to_jsonb($1::text))
                    );
                """
                await conn.execute(sql, item_id, payload_jsonb)

            return {
                "status": "success",
                "message": "문항 수정이 완료되었습니다.",
                "item_id": item_id,
            }

        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON 문항 수정 중 오류: {str(e)}")
    

#문제은행 문항 등록 API
@router.post("/item/enrollment/bank")
async def enroll_items_to_bank(request: Request):
    body = await request.json()

    item_id: Optional[str] = body.get("item_id")
    ins_user_id: Optional[str] = body.get("user_id")

    if not item_id:
        raise HTTPException(status_code=400, detail="item_id는 필수입니다.")
    if not ins_user_id:
        raise HTTPException(status_code=400, detail="ins_user_id는 필수입니다.")

    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        async with conn.transaction():

            # 1) 소스(JSON)에서 해당 item 가져오기
            src_sql = """
                SELECT
                    q->>'item_id'                         AS item_id,
                    split_part(q->>'item_id','-',2)       AS course_id,      -- ITEM-<course_id>-<suffix>
                    q->>'item_content'                    AS item_content,
                    (q->'item_choices')::json             AS item_choices,   -- 목표 컬럼이 json이면 ::json
                    q->>'item_answer'                     AS item_answer,
                    q->>'item_explain'                    AS item_explain,
                    COALESCE(q->>'item_type_cd','MC')     AS item_type_cd,
                    COALESCE(q->>'item_diff_cd','E')      AS item_diff_cd,
                    q->>'grading_note'                   AS grading_note
                FROM aita_chatbot_session s
                CROSS JOIN LATERAL jsonb_array_elements(s.chat_file_json->'quiz_data'->'questions') AS q
                WHERE q->>'item_id' = $1
                LIMIT 1;
            """
            row = await conn.fetchrow(src_sql, item_id)
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"llm이 생성한 문항에서 item_id를 찾지 못했습니다: {item_id}"
                )

            # 2) 대상 테이블에 존재 여부 확인
            exists = await conn.fetchval(
                "SELECT 1 FROM aita_quiz_item_mst WHERE item_id = $1 LIMIT 1",
                item_id
            )

            if exists:
                # 3-A) UPDATE
                update_sql = """
                    UPDATE aita_quiz_item_mst
                    SET
                        course_id     = $2,
                        item_content  = $3,
                        item_choices  = $4,
                        item_answer   = $5,
                        item_explain  = $6,
                        item_type_cd  = $7,
                        item_diff_cd  = $8,
                        upd_user_id   = $9,
                        upd_dt        = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul',
                        grading_note  = $10
                    WHERE item_id = $1
                    RETURNING item_id;
                """
                updated_id = await conn.fetchval(
                    update_sql,
                    row["item_id"],
                    row["course_id"],
                    row["item_content"],
                    row["item_choices"],
                    row["item_answer"],
                    row["item_explain"],
                    row["item_type_cd"],
                    row["item_diff_cd"],
                    ins_user_id,
                    row["grading_note"]
                )
                return {
                    "status": "updated",
                    "message": "문제은행 UPDATE 완료",
                    "item_id": updated_id
                }

            else:
                # 3-B) INSERT (신규)
                insert_sql = """
                    INSERT INTO aita_quiz_item_mst
                        (item_id, course_id, item_content, item_choices, item_answer, item_explain,
                         item_type_cd, item_diff_cd, ins_user_id, ins_dt, del_yn, grading_note)
                    VALUES
                        ($1, $2, $3, $4, $5, $6,
                         $7, $8, $9, (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'), 'N', $10)
                    RETURNING item_id;
                """
                inserted_id = await conn.fetchval(
                    insert_sql,
                    row["item_id"],
                    row["course_id"],
                    row["item_content"],
                    row["item_choices"],
                    row["item_answer"],
                    row["item_explain"],
                    row["item_type_cd"],
                    row["item_diff_cd"],
                    ins_user_id,
                    row["grading_note"]
                )
                return {
                    "status": "inserted",
                    "message": "문제은행 INSERT 완료",
                    "item_id": inserted_id
                }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제은행 문항 등록 중 오류: {str(e)}")
    finally:
        await conn.close()

#문제은행 문항 조회 API 
@router.post("/item/data/bank")
async def item_data_bank(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    item_diff_cd = body.get("item_diff_cd", [])
    item_type_cd = body.get("item_type_cd", [])
    ins_dt = body.get("ins_dt")
    upd_dt = body.get("upd_dt")

    if not cls_id or not item_diff_cd or not item_type_cd:
        raise HTTPException(status_code=400, detail="cls_id, item_diff_cd, and item_type_cd are required")

    # 기본 쿼리
    query_parts = [
        "SELECT * FROM aita_quiz_item_mst",
        "WHERE course_id = $1",
        "AND del_yn = 'N'",
        "AND item_diff_cd = ANY($2::text[])",
        "AND item_type_cd = ANY($3::text[])"
    ]
    
    # 파라미터 리스트
    params = [None, item_diff_cd, item_type_cd] # index 0은 placeholder, 실제로는 course_id가 $1로 들어감 (아래에서 처리)
    
    param_idx = 4 # $1, $2, $3은 이미 고정

    # ins_dt가 있을 때만 조건 추가
    if ins_dt:
        query_parts.append(f"AND ins_dt = ANY(${param_idx}::timestamp[])")
        params.append(ins_dt)
        param_idx += 1
        
    # upd_dt가 있을 때만 조건 추가
    if upd_dt:
        query_parts.append(f"AND upd_dt = ANY(${param_idx}::timestamp[])")
        params.append(upd_dt)
        param_idx += 1

    final_query = "\n".join(query_parts)
        
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            course_row = await conn.fetchrow("SELECT course_id FROM cls_mst WHERE cls_id = $1", cls_id)
            if not course_row:
                raise HTTPException(status_code=404, detail="cls_id not found")
            course_id = course_row["course_id"]

            # params의 첫 번째 요소(None)을 course_id로 교체하거나 args unpacking 사용
            # asyncpg fetch는 인자를 *args로 받으므로 리스트를 unpacking 해야 함
            
            # 최종 실행 인자 구성: course_id + 나머지 동적 파라미터들
            execution_args = [course_id, item_diff_cd, item_type_cd]
            if ins_dt: execution_args.append(ins_dt)
            if upd_dt: execution_args.append(upd_dt)

            rows = await conn.fetch(final_query, *execution_args)

            return [dict(row) for row in rows] # 간단하게 row 변환

        finally:
            await conn.close()
    except Exception as e:
        print(f"Error Query: {final_query}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    

@router.post("/item/search/suffix")
async def search_item_by_suffix(request: Request):
    """
    문항 식별자(item_id)의 뒤 6자리(Suffix)를 이용한 문항 조회 API
    옵션: courseId, ins_user_id, upd_user_id
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 1. 필수 파라미터 추출 및 검증
    item_suffix = body.get("item_suffix")
    
    if not item_suffix:
        raise HTTPException(status_code=400, detail="item_suffix is required")
    
    # 6자리 검증 (DB 부하를 줄이기 위해 애플리케이션 단에서 먼저 체크)
    if len(item_suffix) != 6:
        raise HTTPException(status_code=400, detail="item_suffix must be exactly 6 characters")

    # 2. 옵션 파라미터 추출
    # 요청은 camelCase(courseId)로 오지만, DB 조회 등 로직에서는 snake_case 변수로 관리
    course_id = body.get("courseId") 
    ins_user_id = body.get("ins_user_id")
    upd_user_id = body.get("upd_user_id")

    # 3. 쿼리 구성 (정규식 대신 RIGHT 함수 사용)
    # RIGHT(item_id, 6) : item_id 컬럼의 오른쪽 끝에서 6글자를 추출
    query_parts = [
        "SELECT * FROM aita_quiz_item_mst",
        "WHERE del_yn = 'N'",
        "AND RIGHT(item_id, 6) = $1" 
    ]
    
    # 파라미터 리스트 초기화 (첫 번째 파라미터는 무조건 item_suffix)
    execution_args = [item_suffix]
    param_idx = 2 # $1은 사용했으므로 $2부터 시작

    # 4. 동적 조건 추가
    if course_id:
        query_parts.append(f"AND course_id = ${param_idx}")
        execution_args.append(course_id)
        param_idx += 1
    
    if ins_user_id:
        query_parts.append(f"AND ins_user_id = ${param_idx}")
        execution_args.append(ins_user_id)
        param_idx += 1
        
    if upd_user_id:
        query_parts.append(f"AND upd_user_id = ${param_idx}")
        execution_args.append(upd_user_id)
        param_idx += 1

    final_query = "\n".join(query_parts)

    # 5. DB 실행
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 쿼리 실행
            rows = await conn.fetch(final_query, *execution_args)

            # # 결과가 없는 경우 404 처리 (요구사항에 따라 빈 리스트 반환도 가능하지만, 명시적 에러 요청 반영)
            # if not rows:
            #     raise HTTPException(status_code=404, detail=f"No item found with suffix '{item_suffix}' and given conditions.")

            # 결과 반환 (dict 변환)
            # return [dict(row) for row in rows]
            return {"items": [dict(row) for row in rows]}     
        
        finally:
            await conn.close()

    except HTTPException as he:
        # 위에서 발생시킨 HTTP 에러는 그대로 전달
        raise he
    except Exception as e:
        # DB 연결 실패, SQL 문법 오류 등 예측하지 못한 에러
        print(f"Error Query: {final_query}")
        print(f"Error Args: {execution_args}")
        raise HTTPException(status_code=500, detail=f"Database processing error: {str(e)}")
    

    
# 문제은행 문항 수정 API
@router.post("/item/edit/bank")
async def item_edit_bank(request: Request):
    try:
        body = await request.json()
        user_id:str = body.get("user_id")
        item_id: str = body.get("item_id")
        item_content: str = body.get("item_content", "")
        item_choices: str = body.get("item_choices")
        item_answer: str = body.get("item_answer", "")
        item_explain: str = body.get("item_explain", "")
        item_diff_cd: str = body.get("item_diff_cd", "E")
        grading_note: str = body.get("grading_note")

        # 유효성 검사
        if not item_id:
            raise HTTPException(
                status_code=400,
                detail="item_id는 필수입니다."
            )
        
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:

            update_query = """
                UPDATE aita_quiz_item_mst
                SET 
                    item_content = $2,
                    item_choices = $3::json,
                    item_answer = $4,
                    item_explain = $5,
                    item_diff_cd = $6,
                    upd_user_id = $7,
                    upd_dt = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul',
                    grading_note = $8
                WHERE item_id = $1;
            """

            await conn.execute(
                update_query,
                item_id,
                item_content,
                json.dumps(item_choices, ensure_ascii=False) if item_choices is not None else None,
                item_answer,
                item_explain,
                item_diff_cd,
                user_id,
                grading_note
            )

        finally:
            await conn.close()

        return {
            "status": "success",
            "message": "문제은행 문항 수정이 완료되었습니다.",
            "item_id": item_id
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문제은행 수정 중 오류 발생: {str(e)}"
        )
    
# 문제은행 문항 삭제 API
@router.post("/item/delete/bank")    
async def item_delete_bank(request: Request):
    body = await request.json()
    item_ids = body.get("item_ids")  # 리스트로 받음

    if not item_ids or not isinstance(item_ids, list):
        raise HTTPException(status_code=400, detail="item_ids must be a non-empty list")

    query = """
        UPDATE aita_quiz_item_mst
        SET del_yn = 'Y', upd_dt = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Seoul'
        WHERE item_id = ANY($1::text[])
        RETURNING item_id;
    """

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            rows = await conn.fetch(query, item_ids)
            deleted_ids = [r["item_id"] for r in rows]

            return {
                "status": "success",
                "message": "문제은행 문항이 삭제되었습니다.",
                "deleted_item_ids": deleted_ids
            }

        finally:
            await conn.close()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문제은행 문항 삭제 중 오류 발생: {str(e)}"
        )
    

#시험지 리스트 조회 API
@router.post("/exam/list")
async def exam_list(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")

    if not cls_id:
        raise HTTPException(status_code=400, detail="cls_id is required")

    query = """
        SELECT * FROM aita_quiz_cls_map WHERE cls_id = $1 AND del_yn = 'N';
    """

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            rows = await conn.fetch(query, cls_id)
            exam_list = [
                {
                    "quiz_id": row["quiz_id"],
                    "quiz_nm": row["quiz_nm"],
                    "ins_dt": row["ins_dt"],
                    "cls_id": row["cls_id"],
                }
                for row in rows
            ]

            return {
                "count": len(exam_list),  # ✅ 결과 개수
                "exam_list": exam_list        
            }
        finally:
            await conn.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시험지 리스트 조회 중 오류가 발생했습니다: {str(e)}")
    
#시험지 삭제 API
@router.post("/exam/delete")
async def exam_delete(request: Request):
    body = await request.json()
    quiz_id = body.get("quiz_id")

    if not quiz_id:
        raise HTTPException(status_code=400, detail="quiz_id is required")

    query = """
        UPDATE aita_quiz_cls_map
        SET del_yn = 'Y'
        WHERE quiz_id = $1
    """

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            await conn.execute(query, quiz_id)

            return {"status": "deleted", "quiz_id": quiz_id}
        finally:
            await conn.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시험지 삭제 중 오류가 발생했습니다: {str(e)}")
    
#시험지 상세 조회 API
@router.post("/exam/detail")
async def exam_detail(request: Request):
    body = await request.json()
    quiz_id = body.get("quiz_id")

    if not quiz_id:
        raise HTTPException(status_code=400, detail="quiz_id is required")

    query_items = """
        SELECT qms.item_id
            ,qms.item_content --문항 
            ,qms.item_choices --보기 
            ,qms.item_answer  --정답 
            ,qms.item_explain --해설
            ,qms.item_type_cd --유형
            ,qms.item_diff_cd --난이도
            ,qms.grading_note --채점기준
        FROM aita_quiz_item_map qma
        INNER JOIN aita_quiz_item_mst qms ON qma.item_id = qms.item_id 
        WHERE qma.quiz_id = $1
        ORDER BY qma.order_no
        ;
    """
    query_quiz = """
        SELECT *
        FROM aita_quiz_cls_map
        WHERE quiz_id = $1
        ;
    """

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 아이템 리스트 조회 (여러행)
            rows = await conn.fetch(query_items, quiz_id)
            item_list = [
                {
                    "item_id": row["item_id"],
                    "item_content": row["item_content"],
                    "item_choices": row["item_choices"],
                    "item_answer": row["item_answer"],
                    "item_explain": row["item_explain"],
                    "item_type_cd": row["item_type_cd"],
                    "item_diff_cd": row["item_diff_cd"],
                    "grading_note": row["grading_note"]
                }
                for row in rows
            ]

            quiz_row = await conn.fetchrow(query_quiz, quiz_id)
            if not quiz_row:
                raise HTTPException(status_code=404, detail="quiz_id not found")

            return {
                "quiz_id": quiz_id,
                "quiz_nm": quiz_row["quiz_nm"],
                "count": len(item_list),
                "item_list": item_list
            }
        finally:
            await conn.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시험지 상세 조회 중 오류가 발생했습니다: {str(e)}")
    
#시험지 생성 API
@router.post("/exam/create")
async def exam_create(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    quiz_nm = body.get("quiz_nm")
    items = body.get("items")

    if not cls_id or not quiz_nm or not items or not isinstance(items, list):
        raise HTTPException(status_code=400, detail="cls_id, quiz_nm, items are required")

    # 현재 시간 (KST, offset-naive)
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst).replace(tzinfo=None)
    hash_kst = datetime.now(kst).strftime("%Y%m%d%H%M%S")
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 1. course_id 조회
            course_row = await conn.fetchrow("SELECT course_id FROM cls_mst WHERE cls_id = $1", cls_id)
            if not course_row:
                raise HTTPException(status_code=404, detail="cls_id not found")
            course_id = course_row["course_id"]

            # 2. quiz_id 생성
            value = quiz_nm + course_id + cls_id + hash_kst
            hash_value = hashlib.sha256(value.encode()).hexdigest()[:6]
            quiz_id = f"QUIZ-{course_id}-{hash_value}"

            # 3. aita_quiz_item_map INSERT
            item_values = [(quiz_id, item["item_id"], item["order_no"], now_kst, item["score"]) for item in items]
            item_query = """
                INSERT INTO aita_quiz_item_map (quiz_id, item_id, order_no, ins_dt, score)
                VALUES {}
            """.format(", ".join([f"(${i*5+1}, ${i*5+2}, ${i*5+3}, ${i*5+4}, ${i*5+5})" for i in range(len(item_values))]))
            flat_item_values = [v for row in item_values for v in row]
            await conn.execute(item_query, *flat_item_values)

            # 4. aita_quiz_cls_map INSERT
            await conn.execute(
                "INSERT INTO aita_quiz_cls_map (cls_id, quiz_id, ins_dt, quiz_nm, del_yn) VALUES ($1, $2, $3, $4,'N')",
                cls_id, quiz_id, now_kst, quiz_nm
            )

            # 5. quiz item count
            result = await conn.fetchval("SELECT count(*) FROM aita_quiz_item_map WHERE quiz_id = $1", quiz_id)

            return {
                "status": "success",
                "quiz_id": quiz_id,
                "quiz_nm": quiz_nm,
                "cls_id": cls_id,
                "item_count": result
            }

        finally:
            await conn.close()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시험지 생성 중 오류가 발생했습니다: {str(e)}")
    

#답안 채점 근거 API
@router.post("/item/scoring/rationale")
async def item_scoring_rationale(request: Request):
    try:
        body = await request.json()
        
        # RunScoringAgentTeam 함수를 비동기적으로 호출
        result_dict = await RunScoringAgentTeam(body)
        
        return result_dict
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채점 중 오류가 발생했습니다: {str(e)}"
        )
    

#OCR API
@router.post("/ocr")
async def ocr(request: Request):
    try:
        body = await request.json()

        # 필수 필드 검증
        img_path = body.get("img_path")
        if not img_path:
            return JSONResponse(
                status_code=400,
                content={
                    "result": "fail",
                    "detail": "OCR 이미지경로(img_path)는 필수 입력값입니다."
                }
            )

        # 선택적 OCR 설정
        ocr_config = body.get("ocr_config", {})

        # OCR 실행
        result_ocr = await ocr_main(img_path=img_path, ocr_config=ocr_config)

        return {
            "result": "ok",
            "data": result_ocr
        }

    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "result": "fail",
                "detail": e.detail
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "result": "fail",
                "detail": f"OCR 처리 중 내부 오류가 발생했습니다: {str(e)}"
            }
        )

#문제은행 문항 직접생성
@router.post("/item/manual/bank")
async def item_manual_bank(request: Request):
    try:
        body = await request.json()
        questions = body.get("questions", [])
        cls_id = body.get("cls_id")
        ins_user_id = body.get("ins_user_id")

        # 유효성 검사
        if not questions or not isinstance(questions, list):
            raise HTTPException(
                status_code=400,
                detail="questions는 필수입니다."
            )

        if not cls_id:
            raise HTTPException(
                status_code=400,
                detail="cls_id는 필수입니다."
            )

        if not ins_user_id:
            raise HTTPException(
                status_code=400,
                detail="ins_user_id는 필수입니다."
            )
        # 현재 시간 (KST, offset-naive)
        kst = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst).replace(tzinfo=None)
        # hash_kst = datetime.now(kst).strftime("%Y%m%d%H%M%S")
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            for question in questions:
                # item_id 생성: ITEM-course_id-해시값6자리
                course_row = await conn.fetchrow("SELECT course_id, user_id FROM cls_mst WHERE cls_id = $1", cls_id)
                if not course_row:
                    raise HTTPException(status_code=404, detail="cls_id not found")
                course_id = course_row["course_id"]
                item_content = question.get("question", "")
                # value = course_id + item_content + question.get('type', '') + hash_kst
                # hash_value = hashlib.sha256(value.encode()).hexdigest()[:6]
                # item_id = f"ITEM-{course_id}-{hash_value}"
                rand_str = ''.join(random.choices(string.ascii_uppercase, k=6))
                item_id = f"ITEM-{course_id}-{rand_str}"

                # item_choices 처리
                item_choices = None
                if question.get("type") == "MC" and question.get("choices"):
                    choices_dict = {}
                    choice_count = question.get("choiceCount", 4)  # 프론트엔드에서 전송된 선택지 개수
                    choices = question.get("choices", [])

                    # choiceCount만큼 또는 실제 choices 길이만큼 처리 (최대 6개)
                    actual_count = min(choice_count, len(choices), 6)

                    for i in range(actual_count):
                        if i < len(choices) and choices[i]:  # 빈 선택지는 제외
                            choices_dict[str(i+1)] = choices[i]

                    item_choices = json.dumps(choices_dict, ensure_ascii=False)

                # 데이터 삽입
                insert_query = """
                    INSERT INTO aita_quiz_item_mst (
                        item_id, course_id, item_content, item_choices,
                        item_answer, item_explain, item_type_cd, item_diff_cd,
                        file_path, ins_user_id, ins_dt, upd_user_id,
                        upd_dt, del_yn, grading_note
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10, $11, NULL,
                        NULL, 'N', $12
                    )
                """

                await conn.execute(
                    insert_query,
                    item_id,                                    # $1 item_id
                    course_id,                                  # $2 course_id
                    item_content,                               # $3 item_content
                    item_choices,                               # $4 item_choices
                    question.get("answer", ""),                 # $5 item_answer
                    question.get("explanation", ""),            # $6 item_explain
                    question.get("type", "MC"),                 # $7 item_type_cd
                    question.get("difficulty", "E"),            # $8 item_diff_cd
                    "",                                         # $9 file_path
                    ins_user_id,                                # $10 ins_user_id
                    now_kst,                                    # $11 ins_dt
                    question.get("grading_note")                # $12 grading_note
                )

        finally:
            await conn.close()

        return {
            "message": f"{len(questions)}개 문항({item_id} 포함)이 문제은행에 등록되었습니다."
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문제은행 등록 중 오류 발생: {str(e)}"
        )
