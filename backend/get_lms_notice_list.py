import requests
import sys
import os
import hashlib
import asyncio
import asyncpg
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

# 상위 디렉토리 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hallym_smart_lead import slead_course
from hallym_smart_lead import slead_notice
from hallym_smart_lead import slead_member 
from config import DATABASE_CONFIG

KST = timezone(timedelta(hours=9))  # 한국 시간대 (UTC+9)

async def init_db_pool():
    """비동기 PostgreSQL 연결 풀 초기화"""
    return await asyncpg.create_pool(**DATABASE_CONFIG)

def generate_notice_file_id(cls_id, notice_url, notice_num):
    """cls_id, notice_url, notice_num을 조합하여 고유한 file_id 생성"""
    # 해시 생성을 위한 문자열 조합
    combined = f"{cls_id}_{notice_url}_{notice_num}"
    # SHA-256 해시 생성 후 앞 6자리 추출
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:6]
    # cls_id, notice_num, 해시값을 조합하여 file_id 생성
    file_id = f"{cls_id}-notice-{notice_num}-{hash_value}"
    return file_id

async def get_cls_id(pool, user_id, course_id, year, semester, cls_sec):
    """cls_id를 가져오는 함수"""
    async with pool.acquire() as conn:
        try:
            select_query = f"""
                SELECT cls_id FROM cls_mst
                WHERE course_id = '{course_id}' AND user_id = '{user_id}' AND cls_yr = '{year}' 
                AND (cls_smt = '{semester}' OR cls_smt = '1') AND cls_sec = '{cls_sec}'
            """
            cls_id_record = await conn.fetchrow(select_query)
            
            if cls_id_record is None:
                print(f"cls_id를 찾을 수 없습니다: course_id={course_id}, user_id={user_id}, cls_yr={year}, cls_smt={semester}, cls_sec={cls_sec}")
                return None
            
            return cls_id_record['cls_id']
        except Exception as e:
            print(f"Error getting cls_id: {e}")
            return None

async def insert_notice(pool, file_id, cls_id, user_id, course_id, page_num, notice_num, notice_nm, notice_url):
    """공지사항 정보를 데이터베이스에 저장하는 함수"""
    async with pool.acquire() as conn:
        try:
            # 데이터 삽입
            insert_query = """
                INSERT INTO lms_file_notice 
                (file_id, cls_id, user_id, course_id, page_num, notice_num, notice_nm, notice_url, ins_dt)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (file_id) DO UPDATE 
                SET notice_nm = $7, notice_url = $8, ins_dt = $9
            """
            ins_dt = datetime.now(KST).replace(tzinfo=None)  # 시간대 제거
            await conn.execute(insert_query, file_id, cls_id, user_id, course_id, page_num, notice_num, notice_nm, notice_url, ins_dt)
            return True
        except Exception as e:
            print(f"Error inserting notice: {e}")
            print(f"Parameters: file_id={file_id}, cls_id={cls_id}, user_id={user_id}, course_id={course_id}, page_num={page_num}, notice_num={notice_num}, notice_nm={notice_nm}, notice_url={notice_url}")
            return False

def login_process(user_id, user_pass):
    """사용자 로그인 처리 함수"""
    login = slead_member.SLeadMember()
    result = login.login(user_id, user_pass)
    
    if not result:
        print("로그인 실패")
        return None
    
    return login

def get_courses(login, year, semester):
    """학위 과정 강좌 조회 함수"""
    result, degree_course_data = slead_course.get_degree_course(login, year, semester)
    
    if not result or len(degree_course_data) == 0:
        print("강좌 정보를 불러올 수 없습니다.")
        return None
    
    print(f"총 {len(degree_course_data)}개의 강좌가 조회되었습니다.\n")
    return degree_course_data

async def delete_course_notices(pool, cls_id):
    """특정 cls_id에 해당하는 모든 공지사항 데이터 삭제"""
    async with pool.acquire() as conn:
        try:
            delete_query = "DELETE FROM lms_file_notice WHERE cls_id = $1"
            result = await conn.execute(delete_query, cls_id)
            print(f"  삭제됨: cls_id={cls_id}에 해당하는 모든 공지사항")
            return True
        except Exception as e:
            print(f"Error deleting notices: {e}")
            return False

async def save_course_notices(pool, login, user_id, course_data, year, semester):
    """강좌별 게시판 목록 조회 및 DB 저장 함수"""
    for course_idx, course_data in enumerate(course_data):
        course_id = course_data["course_id"]
        course_name = course_data.get("course_name", "과목명 없음")
        
        print(f"\n{'='*50}\n강좌 {course_idx+1}: {course_name}\n강좌 ID: {course_id}\n{'='*50}")
        
        # 데이터베이스 작업을 위한 course_id와 cls_sec 추출
        try:
            course_info = course_name.split('(')[-1].rstrip(')')
            db_course_id = course_info.split('-')[0].strip()
            cls_sec = course_info.split('-')[-1].strip()
            print(f"  데이터베이스 작업용 과목 ID: {db_course_id}, 분반: {cls_sec}")
        except Exception as e:
            print(f"  과목 정보 추출 오류: {e}")
            db_course_id = course_id  # 기본값으로 원래 course_id 사용
            cls_sec = "01"  # 기본 분반 설정
        
        # cls_id 가져오기
        cls_id = await get_cls_id(pool, user_id, db_course_id, year, semester, cls_sec)
        if not cls_id:
            print(f"  cls_id를 찾을 수 없어 건너뜁니다: {course_name}")
            continue
            
        # 기존 공지사항 데이터 삭제
        await delete_course_notices(pool, cls_id)
            
        # 페이지 조회 변수 초기화
        page_num = 1
        has_more_pages = True
        total_posts = 0
        
        # 전체 페이지 순차적으로 조회
        while has_more_pages:
            print(f"\n{'-'*10} 페이지 {page_num} {'-'*10}")
            result, data_or_error = slead_notice.get_course_notices(login, course_id, page_num)
            
            if result and len(data_or_error['notice_list']) > 0:
                page_posts = len(data_or_error['notice_list'])
                total_posts += page_posts
                print(f"페이지에서 {page_posts}개의 게시글 처리 중...")
                
                for i, notice in enumerate(data_or_error['notice_list']):
                    notice_title = notice['title']
                    notice_url = notice['post_url']
                    notice_num = notice['number']
                    
                    # notice_num을 정수로 변환
                    try:
                        notice_num = int(notice_num)
                    except (ValueError, TypeError):
                        notice_num = i + 1  # 변환 실패시 인덱스 기반으로 번호 부여
                    
                    # file_id 생성 및 DB 저장
                    file_id = generate_notice_file_id(cls_id, notice_url, notice_num)
                    await insert_notice(pool, file_id, cls_id, user_id, db_course_id, page_num, notice_num, notice_title, notice_url)
                
                # 다음 페이지 준비
                page_num += 1
                print(f"현재까지 총 {total_posts}개의 게시글 처리됨")
            elif result and len(data_or_error['notice_list']) == 0:
                print("게시글이 없습니다.")
                has_more_pages = False
            else:
                print(f"게시글 목록을 불러올 수 없습니다: {data_or_error}")
                has_more_pages = False
        
        print(f"\n총 {total_posts}개의 게시글이 처리되었습니다.")

def display_course_notices(login, course_data):
    """강좌별 게시판 목록 조회 및 출력 함수"""
    for course_idx, course_data in enumerate(course_data):
        course_id = course_data["course_id"]
        course_name = course_data.get("course_name", "과목명 없음")
        
        print(f"\n{'='*50}\n강좌 {course_idx+1}: {course_name}\n강좌 ID: {course_id}\n{'='*50}")
        
        # 페이지 조회 변수 초기화
        page_num = 1
        has_more_pages = True
        total_posts = 0
        
        # 전체 페이지 순차적으로 조회
        while has_more_pages:
            print(f"\n{'-'*10} 페이지 {page_num} {'-'*10}")
            result, data_or_error = slead_notice.get_course_notices(login, course_id, page_num)
            
            if result and len(data_or_error['notice_list']) > 0:
                page_posts = len(data_or_error['notice_list'])
                total_posts += page_posts
                print(f"페이지에서 {page_posts}개의 게시글 발견")
                
                # 다음 페이지 준비
                page_num += 1
                print(f"현재까지 총 {total_posts}개의 게시글 발견됨")
            elif result and len(data_or_error['notice_list']) == 0:
                print("게시글이 없습니다.")
                has_more_pages = False
            else:
                print(f"게시글 목록을 불러올 수 없습니다: {data_or_error}")
                has_more_pages = False
        
        print(f"\n총 {total_posts}개의 게시글이 있습니다.")

async def get_cls_info(pool, cls_id):
    """cls_id로부터 course_id, user_id, year, semester, cls_sec 정보 가져오기"""
    async with pool.acquire() as conn:
        try:
            select_query = f"""
                SELECT course_id, user_id, cls_yr, cls_smt, cls_sec 
                FROM cls_mst
                WHERE cls_id = '{cls_id}'
            """
            record = await conn.fetchrow(select_query)
            
            if record is None:
                print(f"cls_id {cls_id}에 대한 정보를 찾을 수 없습니다.")
                return None
            
            return {
                'course_id': record['course_id'],
                'user_id': record['user_id'],
                'year': record['cls_yr'],
                'semester': record['cls_smt'],
                'cls_sec': record['cls_sec']
            }
        except Exception as e:
            print(f"Error getting cls_info: {e}")
            return None

async def run_professor_test_async(user_id, user_pass, year=None, semester=None, save_to_db=False, target_cls_id=None):
    """교수 테스트 실행 함수 (비동기 버전)"""
    # 로그인
    login = login_process(user_id, user_pass)
    if not login:
        return
    print("run_professor_test_async start")
    # DB 연결 풀 초기화
    pool = await init_db_pool()
    try:
        # cls_id가 주어진 경우 해당 강좌 정보 가져오기
        if target_cls_id:
            print(f"\n특정 강좌(cls_id: {target_cls_id})에 대해서만 처리합니다.")
            
            # cls_id로부터 강좌 정보 가져오기
            cls_info = await get_cls_info(pool, target_cls_id)
            if not cls_info:
                print(f"지정된 cls_id({target_cls_id})에 대한 정보를 찾을 수 없습니다.")
                return
            
            # 가져온 정보 사용
            db_course_id = cls_info['course_id']
            cls_sec = cls_info['cls_sec']
            
            # year와 semester가 지정되지 않은 경우, cls_info에서 가져온 값 사용
            if year is None:
                year = cls_info['year']
            if semester is None:
                semester = cls_info['semester']
                
            print(f"강좌 정보: course_id={db_course_id}, year={year}, semester={semester}, cls_sec={cls_sec}")
            
            # 기존 공지사항 삭제
            if save_to_db:
                await delete_course_notices(pool, target_cls_id)
            
            # 강좌 목록 가져오기
            courses = get_courses(login, year, semester)
            if not courses:
                return
                
            # 강좌 목록에서 해당 course_id 찾기
            found = False
            for course_data in courses:
                course_id = course_data["course_id"]
                course_name = course_data.get("course_name", "과목명 없음")
                
                try:
                    course_info = course_name.split('(')[-1].rstrip(')')
                    extracted_db_course_id = course_info.split('-')[0].strip()
                    
                    # 일치하는 강좌 찾기
                    if extracted_db_course_id == db_course_id:
                        print(f"\n대상 강좌를 찾았습니다: {course_name}")
                        
                        if save_to_db:
                            # 해당 강좌만 처리
                            await process_single_course(pool, login, user_id, course_data, db_course_id, cls_sec, target_cls_id, year, semester)
                        else:
                            # 단순 출력
                            display_single_course_notices(login, course_data)
                        
                        found = True
                        break
                except Exception as e:
                    print(f"  강좌 정보 처리 중 오류 발생: {e}")
                    continue
            
            if not found:
                print(f"지정된 cls_id({target_cls_id})와 일치하는 강좌를 찾을 수 없습니다.")
        else:
            # cls_id가 지정되지 않은 경우 모든 강좌 처리
            if year is None or semester is None:
                print("모든 강좌 처리 시에는 year와 semester 값이 필요합니다.")
                return
            
            # 강좌 조회
            courses = get_courses(login, year, semester)
            if not courses:
                return
            
            if save_to_db:
                # 모든 강좌에 대해 처리
                await save_course_notices(pool, login, user_id, courses, year, semester)
            else:
                # 데이터베이스 저장 없이 단순 출력만
                display_course_notices(login, courses)
    finally:
        # 연결 풀 닫기
        await pool.close()

def run_professor_test(user_id, user_pass, year=None, semester=None, save_to_db=False, target_cls_id=None):
    """교수 테스트 실행 함수 (동기 버전 래퍼)"""
    asyncio.run(run_professor_test_async(user_id, user_pass, year, semester, save_to_db, target_cls_id))

async def process_single_course(pool, login, user_id, course_data, db_course_id, cls_sec, cls_id, year, semester):
    """단일 강좌에 대한 공지사항 처리"""
    course_id = course_data["course_id"]
    course_name = course_data.get("course_name", "과목명 없음")
    
    print(f"\n{'='*50}\n강좌: {course_name}\n강좌 ID: {course_id}\ncls_id: {cls_id}\n{'='*50}")
    
    # 페이지 조회 변수 초기화
    page_num = 1
    has_more_pages = True
    total_posts = 0
    
    # 전체 페이지 순차적으로 조회
    while has_more_pages:
        print(f"\n{'-'*10} 페이지 {page_num} {'-'*10}")
        result, data_or_error = slead_notice.get_course_notices(login, course_id, page_num)
        
        if result and len(data_or_error['notice_list']) > 0:
            page_posts = len(data_or_error['notice_list'])
            total_posts += page_posts
            print(f"페이지에서 {page_posts}개의 게시글 처리 중...")
            
            for i, notice in enumerate(data_or_error['notice_list']):
                notice_title = notice['title']
                notice_url = notice['post_url']
                notice_num = notice['number']
                
                # notice_num을 정수로 변환
                try:
                    notice_num = int(notice_num)
                except (ValueError, TypeError):
                    notice_num = i + 1  # 변환 실패시 인덱스 기반으로 번호 부여
                
                # file_id 생성 및 DB 저장
                file_id = generate_notice_file_id(cls_id, notice_url, notice_num)
                await insert_notice(pool, file_id, cls_id, user_id, db_course_id, page_num, notice_num, notice_title, notice_url)
            
            # 다음 페이지 준비
            page_num += 1
            print(f"현재까지 총 {total_posts}개의 게시글 처리됨")
        elif result and len(data_or_error['notice_list']) == 0:
            print("게시글이 없습니다.")
            has_more_pages = False
        else:
            print(f"게시글 목록을 불러올 수 없습니다: {data_or_error}")
            has_more_pages = False
    
    print(f"\n총 {total_posts}개의 게시글이 처리되었습니다.")

def display_single_course_notices(login, course_data):
    """단일 강좌의 게시판 목록 조회 및 출력 함수"""
    course_id = course_data["course_id"]
    course_name = course_data.get("course_name", "과목명 없음")
    
    print(f"\n{'='*50}\n강좌: {course_name}\n강좌 ID: {course_id}\n{'='*50}")
    
    # 페이지 조회 변수 초기화
    page_num = 1
    has_more_pages = True
    total_posts = 0
    
    # 전체 페이지 순차적으로 조회
    while has_more_pages:
        print(f"\n{'-'*10} 페이지 {page_num} {'-'*10}")
        result, data_or_error = slead_notice.get_course_notices(login, course_id, page_num)
        
        if result and len(data_or_error['notice_list']) > 0:
            page_posts = len(data_or_error['notice_list'])
            total_posts += page_posts
            print(f"페이지에서 {page_posts}개의 게시글 발견")
            
            # 다음 페이지 준비
            page_num += 1
            print(f"현재까지 총 {total_posts}개의 게시글 발견됨")
        elif result and len(data_or_error['notice_list']) == 0:
            print("게시글이 없습니다.")
            has_more_pages = False
        else:
            print(f"게시글 목록을 불러올 수 없습니다: {data_or_error}")
            has_more_pages = False
    
    print(f"\n총 {total_posts}개의 게시글이 있습니다.")

if __name__ == "__main__":
    # professor    
    input_user_id_pr = '45932'
    input_user_pass_pr = 'abcde!234'
    input_cls_id = "2024-2-17156-01"  # 특정 cls_id만 처리

    # 전체 프로세스 실행
    run_professor_test(input_user_id_pr, input_user_pass_pr, None, None, True, input_cls_id)

