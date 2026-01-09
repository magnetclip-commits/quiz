from fastapi import APIRouter
from config import DATABASE_CONFIG
from fastapi import Request, HTTPException
import asyncpg



router = APIRouter()


# 수강학생 목록 조회 API
@router.post("/enrollment/student")
async def enrollment_student(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")

    if not cls_id:
        raise HTTPException(status_code=400, detail="cls_id is required")

    query = """
        SELECT ce.user_id 
            ,um.user_nm 
            ,um.org_nm 
            ,um.dpt_nm 
            ,COALESCE(cs.session_count,0) AS session_cnt
            ,COALESCE(cs.total_seq_count,0) AS tot_seq_cnt
        FROM cls_enr ce
        LEFT OUTER JOIN user_mst um ON ce.user_id = um.user_id
        LEFT OUTER JOIN (SELECT user_id
                            ,cls_id
                            ,COUNT(session_id) AS session_count
                            ,SUM(jsonb_array_length(chat_content)) AS total_seq_count
                        FROM chatbot_session
                        GROUP BY user_id,cls_id
                        ) cs ON ce.user_id =cs.user_id AND ce.cls_id = cs.cls_id
        WHERE ce.cls_id = $1
        AND ce.user_div = 'S'
        -- AND um.user_ctg <> 'TEST'
        ORDER BY 1
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, cls_id)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"user_id": row["user_id"],
                     "user_nm": row["user_nm"],
                     "dpt_nm": row["dpt_nm"],
                     "org_nm": row["org_nm"],
                     "session_cnt": row["session_cnt"],
                     "tot_seq_cnt": row["tot_seq_cnt"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# 수강학생 정보 조회 API
@router.post("/enrolled/student/chatinfo")
async def enrolled_student_info(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    cls_id = body.get("cls_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id and cls_id is required")

    query = """
        SELECT *
        FROM v_chatbot_session
        WHERE user_id = $1
        AND cls_id = $2
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, user_id, cls_id)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"user_id": row["user_id"],
                     "session_title": row["session_title"],
                     "chat_start_dt": row["chat_start_dt"],
                     "answer_dt": row["answer_dt"],
                     "seq": row["seq"],
                     "seq_question": row["seq_question"],
                     "seq_response": row["seq_response"],
                     "positive_yn": row["positive_yn"],
                     "positive_selected_options_label": row["positive_selected_options_label"],
                     "positive_feedback": row["positive_feedback"],
                     "negative_yn": row["negative_yn"],
                     "negative_selected_options_label": row["negative_selected_options_label"],
                     "negative_feedback": row["negative_feedback"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    


# 과목별 많이 한 질문 조회 API
@router.post("/class/top/question")
async def class_top_question(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")


    query = """
        select * 
        from hltutor.cls_top_qst 
        where cls_id = $1;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, cls_id)

            # 결과를 리스트의 딕셔너리 형태로 변환하여 반환
            return [dict(row) for row in rows]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    



# 수강학생 예복습 횟수 정보 조회 API
@router.post("/enrolled/student/learncount")
async def enrolled_student_learncount(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")

    if not cls_id:
        raise HTTPException(status_code=400, detail="cls_id is required")

    query = """
        SELECT cs.user_id 
            ,um.user_nm 
            ,um.org_nm --소속 
            ,um.dpt_nm --학과 
            ,cs.week_num
            ,SUM(CASE WHEN cs.session_type = 'PRE' THEN COALESCE(cs.session_count,0) ELSE 0 END) AS pre_cnt --예습 횟수 
            ,SUM(CASE WHEN cs.session_type = 'REV' THEN COALESCE(cs.session_count,0) ELSE 0 END) AS rev_cnt --복습 횟수 
        FROM (SELECT session_type
                    ,user_id
                    ,cls_id
                    ,submit_yn
                    ,week_num
                    ,COUNT(session_id) AS session_count
                FROM chatbot_session_learn
                GROUP BY session_type, user_id, cls_id, submit_yn, week_num
            ) cs 
        INNER JOIN user_mst um ON cs.user_id = um.user_id
        WHERE cs.cls_id = $1
        -- AND um.user_div = 'S'	 -- 학생만 
        -- AND um.user_ctg <> 'TEST'
        AND cs.submit_yn = 'Y'	 -- 제출한 사람만 
        GROUP BY cs.user_id, um.user_nm, um.org_nm, um.dpt_nm, cs.week_num
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, cls_id)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"user_id": row["user_id"],
                     "user_nm": row["user_nm"],
                     "org_nm": row["org_nm"],
                     "dpt_nm": row["dpt_nm"],
                     "week_num": row["week_num"],
                     "pre_cnt": row["pre_cnt"],
                     "rev_cnt": row["rev_cnt"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# 수강학생 예복습 횟수 정보 조회 학생버전 API
@router.post("/enrolled/student/learncount/std")
async def enrolled_student_learncount_std(request: Request):
    body = await request.json()
    cls_id = body.get("cls_id")
    user_id = body.get("user_id")

    if not cls_id:
        raise HTTPException(status_code=400, detail="cls_id is required")

    query = """
        SELECT cs.user_id 
            ,um.user_nm 
            ,um.org_nm --소속 
            ,um.dpt_nm --학과 
            ,cs.week_num
            ,SUM(CASE WHEN cs.session_type = 'PRE' THEN COALESCE(cs.session_count,0) ELSE 0 END) AS pre_cnt --예습 횟수 
            ,SUM(CASE WHEN cs.session_type = 'REV' THEN COALESCE(cs.session_count,0) ELSE 0 END) AS rev_cnt --복습 횟수 
        FROM (SELECT session_type
                    ,user_id
                    ,cls_id
                    ,submit_yn
                    ,week_num
                    ,COUNT(session_id) AS session_count
                FROM chatbot_session_learn
                GROUP BY session_type, user_id, cls_id, submit_yn, week_num
            ) cs 
        INNER JOIN user_mst um ON cs.user_id = um.user_id
        WHERE cs.cls_id = $1 AND cs.user_id = $2
        -- AND um.user_div = 'S'	 -- 학생만 
        -- AND um.user_ctg <> 'TEST'
        AND cs.submit_yn = 'Y'	 -- 제출한 사람만 
        GROUP BY cs.user_id, um.user_nm, um.org_nm, um.dpt_nm, cs.week_num
        ;
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, cls_id, user_id)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"user_id": row["user_id"],
                     "user_nm": row["user_nm"],
                     "org_nm": row["org_nm"],
                     "dpt_nm": row["dpt_nm"],
                     "week_num": row["week_num"],
                     "pre_cnt": row["pre_cnt"],
                     "rev_cnt": row["rev_cnt"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# 수강학생 예복습 리스트 조회 API
@router.post("/enrolled/student/learnlist")
async def enrolled_student_learnlist(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    cls_id = body.get("cls_id")
    session_type = body.get("session_type")
    week_num = body.get("week_num")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id and cls_id is required")

    query = """
        SELECT session_id, week_num, submit_dt, count(DISTINCT question_key) AS question_cnt, count(turn_id) AS turn_cnt
        FROM v_chatbot_session_learn
        WHERE cls_id = $1 and week_num = $4
        AND user_id = $2
        AND session_type = $3
        GROUP BY session_id, week_num, submit_dt
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, cls_id, user_id, session_type, week_num)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {"week_num": row["week_num"],
                     "submit_dt": row["submit_dt"],
                     "session_id": row["session_id"],
                     "question_cnt": row["question_cnt"],
                     "turn_cnt": row["turn_cnt"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# 수강학생 예복습 정보 조회 API
@router.post("/enrolled/student/learninfo")
async def enrolled_student_learninfo(request: Request):
    body = await request.json()
    session_id = body.get("session_id")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    query = """
        SELECT *
        FROM v_chatbot_session_learn
        WHERE session_id = $1;
    """

    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            rows = await conn.fetch(query, session_id)

            comment_row = await conn.fetchrow(
                "SELECT rev_comment FROM hltutor.chatbot_session_learn WHERE session_id = $1;",
                session_id
            )
            comment = comment_row["rev_comment"] if comment_row else ""

            learn_info = [
                {
                    "question_key": row["question_key"],
                    "turn_id": row["turn_id"],
                    "question": row["question"],
                    "week_num": row["week_num"],
                    "submit_dt": row["submit_dt"],
                    "response_lead": row["response_lead"],
                }
                for row in rows
            ]

            return {
                "comment": comment,
                "learn_info": learn_info
            }

        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    

# 주차 정보 조회
@router.post("/year/smt/info")
async def year_smt_info(request: Request):
    body = await request.json()
    year = body.get("year")
    semester = body.get("semester")

    if not year:
        raise HTTPException(status_code=400, detail="session_id is required")

    query = """
        SELECT *
        FROM comm_smt_week csw 
        WHERE yr = $1
        AND smt = $2
    """

    try:
        # PostgreSQL 비동기 연결
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        try:
            # 여러 행을 가져오기 위해 fetch() 사용
            rows = await conn.fetch(query, year, semester)

            # 결과가 있다면 리스트로 변환하여 반환, 없다면 빈 리스트 반환
            return [
                    {
                     "week_num": row["week_num"],
                     "week_full_nm": row["week_full_nm"],
                     }
                for row in rows
            ]
        finally:
            await conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")