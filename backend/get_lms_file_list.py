'''
@modified: 
2025.09.03 '주차' -> '주', '주차'
'''
import sys
import os
from bs4 import BeautifulSoup
import requests
import asyncio
import asyncpg
import hashlib
from datetime import datetime, timedelta, timezone
import re

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallym_smart_lead.slead_member import SLeadMember
from hallym_smart_lead import slead_course
#from hallym_smart_lead.slead_member import SLeadMember
#from hallym_smart_lead.hallym_smart_lead import slead_course
from config import DATABASE_CONFIG

KST = timezone(timedelta(hours=9))  # 한국 시간대 (UTC+9)

async def init_db_pool():
    """비동기 PostgreSQL 연결 풀 초기화"""
    return await asyncpg.create_pool(**DATABASE_CONFIG)

def generate_file_id(cls_id, material_id, material_url):
    """cls_id, material_id, material_url을 조합하여 고유한 file_id 생성"""
    # 해시 생성을 위한 문자열 조합
    combined = f"{cls_id}_{material_id}_{material_url}"
    # SHA-256 해시 생성 후 앞 6자리 추출
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:6]
    # cls_id, material_id, 해시값을 조합하여 file_id 생성
    file_id = f"{cls_id}-{material_id}-{hash_value}"
    return file_id

async def insert_materials(pool, user_id, year, semester, course_id, cls_sec, week_num, week_nm, material_id, material_nm, material_url, material_type):
    """ 강의 자료 정보를 PostgreSQL에 저장하는 함수 """
    async with pool.acquire() as conn:
        try:
            # cls_id 가져오기 ## 학기 변경 전까지 cls_smt = '1' 조건 넣기 
            select_query = f"""
                SELECT cls_id FROM cls_mst
                WHERE course_id = '{course_id}' AND user_id = '{user_id}' AND cls_yr = '{year}' AND (cls_smt = '{semester}' OR cls_smt = '1') AND cls_sec = '{cls_sec}'
            """
            cls_id_record = await conn.fetchrow(select_query)
            if cls_id_record is None:
                print(f"cls_id를 찾을 수 없습니다: course_id={course_id}, user_id={user_id}, cls_yr={year}, cls_smt={semester}, cls_sec={cls_sec}")
                return None
            cls_id = cls_id_record['cls_id']
            print(cls_id)
            # file_id 생성
            file_id = generate_file_id(cls_id, material_id, material_url)

            # 데이터 삽입
            insert_query = """
                INSERT INTO lms_file_weekly (file_id, cls_id, user_id, course_id, week_num, material_id, material_nm, material_url, material_type, ins_dt)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            ins_dt = datetime.now(KST).replace(tzinfo=None)  # 시간대 제거
            await conn.execute(insert_query, file_id, cls_id, user_id, course_id, week_num, material_id, material_nm, material_url, material_type, ins_dt)
            # print(f"Inserted: {material_nm} (file_id: {file_id})")
            return cls_id
        except Exception as e:
            print(f"Error inserting data: {e}")
            print(f"Parameters: cls_id={cls_id}, user_id={user_id}, course_id={course_id}, week_num={week_num}, material_id={material_id}, material_nm={material_nm}, material_url={material_url}, material_type={material_type}")
            return None

async def process_materials(pool, user_id, year, semester, db_course_id, cls_sec, week_num, week_nm, material_id, material_name, material_url, material_type, processed_cls_ids):
    """ 강의 자료를 처리하고 데이터베이스에 저장하는 함수 """
    try:
        cls_id = await insert_materials(pool, user_id, year, semester, db_course_id, cls_sec, week_num, week_nm, material_id, material_name, material_url, material_type)
        if cls_id and cls_id not in processed_cls_ids:
            processed_cls_ids.add(cls_id)
    except Exception as e:
        print(f"Error processing material: {e}")

async def delete_cls_materials(pool, cls_id):
    """ cls_id에 해당하는 모든 자료를 삭제하는 함수 """
    async with pool.acquire() as conn:
        try:
            delete_query = "DELETE FROM lms_file_weekly WHERE cls_id = $1"
            result = await conn.execute(delete_query, cls_id)
            print(f"Deleted all materials for cls_id: {cls_id}, Result: {result}")
        except Exception as e:
            print(f"Error deleting materials: {e}")

async def get_cls_id(pool, user_id, year, semester, course_id, cls_sec):
    """cls_id를 가져오는 함수"""
    async with pool.acquire() as conn:
        try:
            select_query = f"""
                SELECT cls_id FROM cls_mst
                WHERE course_id = '{course_id}' AND user_id = '{user_id}' AND cls_yr = '{year}' AND (cls_smt = '{semester}' OR cls_smt = '1') AND cls_sec = '{cls_sec}'
            """
            cls_id_record = await conn.fetchrow(select_query)
            
            if cls_id_record is None:
                print(f"cls_id를 찾을 수 없습니다: course_id={course_id}, user_id={user_id}, cls_yr={year}, cls_smt={semester}, cls_sec={cls_sec}")
                return None
            
            return cls_id_record['cls_id']
        except Exception as e:
            print(f"Error getting cls_id: {e}")
            return None

async def lms_file_list_main(input_user_id_pr: str, input_user_pass_pr: str, year: str, semester: str, cls_id: str):
    slead_member = SLeadMember()
    print(input_user_id_pr)
    # print(input_user_pass_pr)
    # print(year)
    # print(semester)
    print(cls_id)
    # 로그인
    result = slead_member.login(input_user_id_pr, input_user_pass_pr)
    if not result:
        print("로그인 실패")
        return

    # PostgreSQL 연결 풀 초기화
    pool = await init_db_pool()

    # cls_id로 강좌 정보 조회
    async with pool.acquire() as conn:
        try:
            select_query = f"""
                SELECT course_id, cls_sec FROM cls_mst
                WHERE cls_id = '{cls_id}' AND user_id = '{input_user_id_pr}'
            """
            course_record = await conn.fetchrow(select_query)
            
            if course_record is None:
                print(f"해당 cls_id({cls_id})에 대한 강좌 정보를 찾을 수 없습니다.")
                return
            
            db_course_id = course_record['course_id']
            cls_sec = course_record['cls_sec']
        except Exception as e:
            print(f"강좌 정보 조회 중 오류 발생: {e}")
            return

    # 교과 강좌 목록 조회 
    result = slead_course.get_courses_with_grades(slead_member, year, semester)
    if not result["status"]:
        print(f"강좌 목록 조회 실패: {result['error_msg']}")
        return

    list_course_data = result["data"]
    print("=== 파일 목록 ===")

    # 각 강좌 처리
    for course_data in list_course_data:
        course_name = course_data["course_name"]
        
        # 데이터베이스 작업을 위한 course_id 추출
        current_db_course_id = course_name.split('(')[-1].split('-')[0].strip()
        current_cls_sec = course_name.split('(')[-1].split('-')[-1].strip(')')
        
        # 입력받은 cls_id에 해당하는 강좌만 처리
        if current_db_course_id != db_course_id or current_cls_sec != cls_sec:
            continue

        print(f"\n[강좌] {course_name}")

        # 파일 목록에서는 원래 course_id 사용
        original_course_id = course_data["course_id"]

        # 기존 데이터 삭제 - 먼저 삭제하고 나서 새 데이터 삽입
        await delete_cls_materials(pool, cls_id)

        # 강좌의 주차별 학습활동 조회 
        result = slead_course.get_weekly_learning_activities(slead_member, original_course_id)
        if not result["status"]:
            print(f"  학습활동 조회 실패: {result['error_msg']}")
            continue

        for activity_info in result["data"]:
            week_nm = activity_info["name"]
            #week_num = int(activity_info["name"].split('주차')[0])
            m = re.match(r'^\s*(\d+)\s*주(?:차)?\b', week_nm)
            week_num = int(m.group(1)) if m else 0

            lecture_materials = activity_info["lecture_materials"]
            for lecture_material in lecture_materials:
                material_id = lecture_material["id"]
                material_name = lecture_material["name"]
                material_type = lecture_material["type"]
                material_url = f"https://smartlead.hallym.ac.kr/mod/{material_type}/view.php?id={material_id}"

                # 폴더 타입인 경우 폴더 내용물 가져오기
                if material_type == 'folder':
                    folder_url = f"https://smartlead.hallym.ac.kr/mod/folder/view.php?id={material_id}"
                    try:
                        response = slead_member.get_session().get(folder_url, timeout=30)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            file_links = soup.select('a[href*="/pluginfile.php/"]')
                            for link in file_links:
                                file_name = link.text.strip()
                                if file_name:
                                    material_url = link['href']
                                    # 폴더 내 파일에 폴더의 material_id 사용
                                    await insert_materials(pool, input_user_id_pr, year, semester, db_course_id, cls_sec, week_num, week_nm, material_id, file_name, material_url, material_type)
                    except requests.exceptions.RequestException as e:
                        print(f"Network error processing folder: {e}")
                    except Exception as e:
                        print(f"Error processing folder: {e}")
                else:
                    # 폴더가 아닌 경우 기존 방식대로 처리
                    await insert_materials(pool, input_user_id_pr, year, semester, db_course_id, cls_sec, week_num, week_nm, material_id, material_name, material_url, material_type)

    # 연결 풀 닫기
    await pool.close()

# 비동기 실행
if __name__ == "__main__":
    asyncio.run(lms_file_list_main(
        input_user_id_pr="45932",
        input_user_pass_pr="abcde!234",
        year="2024",
        semester="20",
        cls_id="2024-20-903102-2-01"  
    ))