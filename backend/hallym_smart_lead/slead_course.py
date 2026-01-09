import requests
from bs4 import BeautifulSoup
from typing import List,Tuple, Dict, Union
from .slead_member import SLeadMember
import re
from .slead_parser_helper import extract_query_value_from_atag
from . import slead_syllabus 


def get_degree_course( loginUser : SLeadMember, year : str, semester : str ) -> Union[Tuple[bool, List[Dict[str, str]]], Tuple[bool, str]]:
    """
    학년도/학기로 유저의 학위 과정을 조회한다. 
    year : 학년도(2023,2024,2025....)
    semester : 학기(all : 모든학기 , 10 : 1학기 , 11 : 여름학기, 20 : 2학기, 21 : 겨울학기)

    {
        "year" : "" # 연도
        "semester" : "" # 학기
        "course_id" : "" # 강좌 아이디
        "course_name" : "" # 강좌 이름
        "course_type" : "" # 강좌 종류
    }    

    """
    course_list = []

    session = loginUser.get_session() 

    if session is None :
        return False, "none session"

    if semester != "all" and semester != "10" and semester != "11" and semester != "20" and semester != "21" :
        return False, "invalid semester"            

    # https://smartlead.hallym.ac.kr/local/ubion/user/index.php?year=2023&semester=all
    base_url = "https://smartlead.hallym.ac.kr/local/ubion/user/index.php"  
    url = f"{base_url}?year={year}&semester={semester}"

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            # <tbody class="my-course-lists"> 내부의 <tr>안에 강의 관련 정보가 들어가 있다.
            tbody_bs = soup.find('tbody', class_='my-course-lists')

            if not tbody_bs:
                print("'my-course-lists is not found")
            else:
                tr_elements_bs = tbody_bs.find_all('tr')

                #<tr><td class="text-center">2024</td><td class="text-center">2학기</td><td><div class="course-flex course_label_re_02"><span class="badge badge-course">교과(오프라인)</span> <a href="https://smartlead.hallym.ac.kr/course/view.php?id=17904" class="coursefullname">창의코딩-모두의인공지능 (903102-01)</a></div></td></tr>
                for tr in tr_elements_bs:
                    course_data = tr.find_all("td")

                    if len(course_data) >= 3 :
                        course_info = {
                            "year" : "", 
                            "semester" : "", 
                            "course_id" : "",
                            "course_name" : "",
                            "course_type" : ""
                        } 

                        course_info["year"] = course_data[0].text   
                        course_info["semester"] = course_data[1].text

                        # 강좌 타입
                        course_type = ""
                        badge_element = course_data[2].find('span', class_='badge badge-course')

                        if badge_element:
                            course_info["course_type"] = badge_element.text

                        # 강좌 이름
                        a_tag = course_data[2].find('a', class_='coursefullname')

                        if a_tag:
                            course_info["course_name"] = a_tag.text
                            result, course_id = extract_query_value_from_atag(a_tag, "id")

                            if result:
                                course_info["course_id"] = course_id
                                course_list.append(course_info)                   

            return True, course_list
        else:
            return False, f"페이지 요청 실패! 상태 코드: {response.status_code}"
    except (requests.RequestException, TypeError, KeyError) as e:
        return False, f"error: {e}"

def get_active_courses(login_user : SLeadMember) :
    """
    현재 유저가 진행중인 강좌를 리턴 받는다.

    :리턴값: 성공(True,강좌 목록), 실패(False,"에러메시지")\n
    강좌 목록 리스트(리스트가 비었으면 참여 강좌 없음)
    [
        {
            "title" : "", 강좌 이름
            "badge" : "", 강좌 배지 이름
            "prof"  : "", 교수자
            "course_id" : "" 강좌 아이디
        }
    ]
    """
    course_list = []
    session = login_user.get_session() 

    if not session :
        return {"status": False, "error_msg": "none session"}

    base_url = "https://smartlead.hallym.ac.kr/dashboard.php"  
    url = f"{base_url}"

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            # <ul class="my-course-lists coursemos-layout-0"> -> 이요소 안에 강의 목록이 있음
            ul_course_list = soup.find('ul', class_='my-course-lists')

            if ul_course_list :
                # <li class="course_label_ec">
                li_courses = ul_course_list.find_all("li", class_="course_label_ec")

                for li_course in li_courses:
                    course_info = {
                                    "title" : "",
                                    "badge" : "",
                                    "prof"  : "",
                                    "course_id" : ""
                                    }
                    
                    a_course_link = li_course.find("a", class_="course-link")

                    if a_course_link:
                        course_href = a_course_link.get("href")
                        if course_href:
                            match = re.search(r"id=(\d+)", course_href)
                            if match :
                                course_id = match.group(1)
                                course_info["course_id"] = course_id 

                    #<div class="course-title"><h3>에듀테크 test용</h3><span class="prof">ai_edu_pr / 운영자</span></div></div>
                    div_title = li_course.find("div", class_="course-title")

                    if div_title :
                        h3_title = div_title.find("h3")

                        if h3_title:
                            course_info["title"] = h3_title.text

                        span_prof = div_title.find("span", class_="prof")

                        if span_prof:
                            # ai_edu_pr / 운영자 --> ai_edu_pr
                            prof_text = span_prof.text.split(" / ")[0]
                            course_info["prof"] = prof_text

                    # <div class="badge badge-course">H-Square</div>
                    div_badge = li_course.find("div", class_="badge-course")

                    if div_badge:
                        course_info["badge"] = div_badge.text

                    course_list.append( course_info )                               

            return {"status": True, "data": course_list}
        else:
            return {"status": False, "error_msg": f"page request fail!!! status code: {response.status_code}"}
    except (requests.RequestException, TypeError, KeyError) as e:
        return {"status": False, "error_msg": f"error: {e}"}

def get_courses_with_grades(login_user : SLeadMember, year : str, semester : str) -> dict:
    """
    학년도/학기로 학위과정 강좌 목록을 조회한다. 

        Args:
            login_user (SLeadMember): SLeadMember object
            year (str): 학년도 (예: 2023,2024,2025....)
            semester (str) : 학기 코드
                - 'all' : 모든학기 , '10' : 1학기 , '11' : 여름학기, '20' : 2학기, '21' : 겨울학기

        Returns:
            dict: 조회 결과 포함하는 딕셔너리
            - 성공:
                    {
                    "status": True,
                    "data" :
                    {
                    "year" : "",    # 연도
                    "semester" : "", # 학기
                    "course_name" : "", # 강좌명
                    "course_id" : "", # 강좌 아이디
                    "prof"  : "", # 담당교수
                    "grade"  : "", # 성적
                    "percentage_score" : "", # 백분환산점수
                    "final_grade" : "" # 최종성적
                    }
                    }
            - 실패: 
                {"status": False, "error_msg": ""}
    """
    session = login_user.get_session() 

    if session is None :
        return { "status" : False, "error_msg" : "none session"}

    if semester != "all" and semester != "10" and semester != "11" and semester != "20" and semester != "21" :
        return { "status" : False, "error_msg" : "invalid semester"}

    base_url = "https://smartlead.hallym.ac.kr/local/ubion/user/grade.php"  
    url = f"{base_url}?year={year}&semester={semester}"

    result_list = []

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            tag_table = soup.find('table', class_='table-coursemos')

            if tag_table is None :
                return { "status" : False, "error_msg" : "table-coursemos is not found"}

            tag_tbody = soup.find('tbody')

            if tag_tbody is None :
                return { "status" : False, "error_msg" : "tag_tbody is not found"}

            rows_courses = tag_tbody.find_all('tr')

            for rows_course in rows_courses:
                course_info = {
                    "year" : "",    # 연도
                    "semester" : "", # 학기
                    "course_name" : "", # 강좌명
                    "course_id" : "", # 강좌 아이디
                    "prof"  : "", # 담당교수
                    "grade"  : "", # 성적
                    "percentage_score" : "", # 백분환산점수
                    "final_grade" : "" # 최종성적
                }

                course_data = rows_course.find_all("td")

                if len(course_data) >= 8 :
                    course_info["year"] = course_data[0].text
                    course_info["semester"] = course_data[1].text

                    a_tag = course_data[2].find("a")

                    if a_tag:
                        course_info["course_name"] = a_tag.text.strip()
                        result, course_id = extract_query_value_from_atag(a_tag, "id")

                        if result :
                            course_info["course_id"] = course_id

                    course_info["prof"] = course_data[3].text
                    course_info["grade"] = course_data[4].text
                    course_info["percentage_score"] = course_data[5].text
                    course_info["final_grade"] = course_data[6].text

                    result_list.append(course_info)

            return { "status" : True, "data" : result_list }
        else:
            return { "status" : False, "error_msg" : f"session.get fail! status code: {response.status_code}"}
    except (requests.RequestException, TypeError, KeyError) as e:
        return { "status" : False, "error_msg" : f"exception: {e}"}

def get_course_syllabus(login_user : SLeadMember, course_id : str ) -> dict:
    """
    강좌의 강의 계획서 정보를 조회한다.
    
    Args:
        login_user (SLeadMember) :  유저 인스턴스
        course_id (str) : 강좌 아이디

    Returns:
        dict: 조회 결과 포함하는 딕셔너리
        - 성공:
            {
                "status": "success",
                "data": 
                {
                    "title" : "", # 강의 계획서 제목
                    "course_title" : "", # 과목명
                    "course_code" : "", # 교과목번호
                    "section" : "", # 분반
                    "department" : "", # 소속
                    "research_lab" : "", # 연구소
                    "course_type" : "", # 이수구분
                    "lecture_time" : "", # 시간
                    "main_professor" : "", # 대표교수
                    "joint_course" : "", # 합동강좌
                    "lecture_room" : "", # 강의실
                    "credit_hours" : "", # 학점-수업-실습
                    "email" : "", #전자우편
                    "grading_type" : "", # 성적구분
                    "contact_info" : "", # 연락처
                    "office_hours" : "", # 면담가능시간
                    "prerequisite_courses" : "". 선수과목
                    "lecture_style" : # 수업유형
                    { 
                        "lecture_based" : "", # 강의식
                        "discussion_based" : "", # 토의(토론)식
                        "pbl" : "", # PBL(문제기반학습, 프로젝트기반학습)
                        "team_based" : "", # 팀기반학습
                        "collaborative_individual" : "", # 협동/개별학습
                        "lab_practice" : "", # 실험/실습
                        "field_study" : "", # 현장학습
                        "capstone_design" : "", #캡스톤디자인 
                        "other" : "", #기타
                    }
                    "course_objectives" : "", # 핵심역량
                    "hi_five_skills" : 
                    {
                        "function" : "", # 직무 - 기업가형
                        "innovation" : "", # 도전 - 혁신가형
                        "various_skills" : "", # 기술 - 전문가형
                        "exploration" : "", # 탐구 - 연구자원
                    },
                    "competency_info" : 
                    {
                        "core_competencies" : "", #핵심역량
                        "sub_competencies" : "", #하위역량
                    },

                    "learning_objectives" : # 학습목표 및 평가지표
                    {
                        "knowledge" : [], # 지식
                        "skill" : [],     # 기술  
                        "attitude" : [],  # 태도  
                    },

                    "grading_weight" :  # 성적반영 비율
                    {
                        "total": "",          # 총합 
                        "attendance_score": "",  # 출석 점수
                        "assignment_score": "",   # 과제 점수
                        "midterm_score": "",   # 중간고사 점수
                        "final_score": "",   # 기말고사 점수
                        "other": {  # 기타              
                            "score" : "", # 기타 점수
                            "description" : "", #상세내용
                        }               
                    },

                    "attendance_threshold" : "", #출석 미달 기준 사항
                    "teaching_method" : "", #수업 운영 방법
                    "class_policy" : ",", # 수업 규정
                    "textbooks" : # 교재 및 참고도서
                    {
                        "primary_textbook" : "",    # 주교재
                        "secondary_textbook" : "",  # 부교재 및 참고도서
                    },
                    "additional_notes" : "", # 기타사항
                    "accessibility_support" : "", # 장애학생을 위한 학습 및 평가지원 사항
                }

            - 실패: 
                {
                    "status": "failed",
                    "error_msg": ""
                }
    """
    session = login_user.get_session() 

    if session is None :
        return { "status" : "failed", "error_msg": "none session" }

    base_url = "https://smartlead.hallym.ac.kr/local/ubion/setting/syllabus.php"  
    url = f"{base_url}?id={course_id}"

    course_syllabus = {
        "title" : "", # 강의 계획서 제목
        "course_title" : "", # 과목명
        "course_code" : "", # 교과목번호
        "section" : "", # 분반
        "department" : "", # 소속
        "research_lab" : "", # 연구소
        "course_type" : "", # 이수구분
        "lecture_time" : "", # 시간
        "main_professor" : "", # 대표교수
        "joint_course" : "", # 합동강좌
        "lecture_room" : "", # 강의실
        "credit_hours" : "", # 학점-수업-실습
        "email" : "", #전자우편
        "grading_type" : "", # 성적구분
        "contact_info" : "", # 연락처
        "office_hours" : "", # 면담가능시간
        "lecture_overview" : "", # 수업개요
        "prerequisite_courses" : "", # 선수과목
        "lecture_style" : # 수업유형
        {
            "lecture_based" : "", # 강의식
            "discussion_based" : "", # 토의(토론)식
            "pbl" : "", # PBL(문제기반학습, 프로젝트기반학습)
            "team_based" : "", # 팀기반학습
            "collaborative_individual" : "", # 협동/개별학습
            "lab_practice" : "", # 실험/실습
            "field_study" : "", # 현장학습
            "capstone_design" : "", #캡스톤디자인 
            "other" : "", #기타
        },
        "course_objectives" : "", #교과 목표
        "hi_five_skills" : # Hi FIVE전공능력
        {
            "function" : "", # 직무 - 기업가형
            "innovation" : "", # 도전 - 혁신가형
            "various_skills" : "", # 기술 - 전문가형
            "exploration" : "", # 탐구 - 연구자원
        },
        "competency_info" : #역량
        {
            "core_competencies" : "", #핵심역량
            "sub_competencies" : "", #하위역량
        },

        "learning_objectives" : # 학습목표 및 평가지표
        {
            "knowledge" : [], # 지식
            "skill" : [],     # 기술  
            "attitude" : [],  # 태도  
        },

        "grading_weight" :  # 성적반영 비율
        {
            "total": "",          # 총합 
            "attendance_score": "",  # 출석 점수
            "assignment_score": "",   # 과제 점수
            "midterm_score": "",   # 중간고사 점수
            "final_score": "",   # 기말고사 점수
            "other": {  # 기타              
                "score" : "", # 기타 점수
                "description" : "", #상세내용
            }               
        },

        "attendance_threshold" : "", #출석 미달 기준 사항
        "teaching_method" : "", #수업 운영 방법
        "class_policy" : ",", # 수업 규정
        "textbooks" : # 교재 및 참고도서
        {
            "primary_textbook" : "",    # 주교재
            "secondary_textbook" : "",  # 부교재 및 참고도서
        },
        "additional_notes" : "", # 기타사항
        "accessibility_support" : "", # 장애학생을 위한 학습 및 평가지원 사항
    }

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            div_syllabus = soup.find('div', class_='local_syllabus')

            if div_syllabus is None :
                return { "status" : "failed", "error_msg": "local_syllabus is not found" }
            
            div_course_syllabus = div_syllabus.find('div', class_='course_syllabus')

            if div_course_syllabus is None :
                return { "status" : "failed", "error_msg": "course_syllabus is not found" }

            syllabus_title = div_course_syllabus.find('h3')

            if syllabus_title is None:
                return { "status" : "failed", "error_msg": "syllabus_title is not found" }
            
            course_syllabus["title"] = syllabus_title.text

            tag_tbody = None

            #이 부분 <table> 태그의 짝이 안 맞아서 tbody 를 직접 찾았으니 참고
            for sibling in syllabus_title.next_siblings:
                if sibling.name == "tbody":
                    tag_tbody = sibling
                    break

            # 강좌 기본 정보를 추출한다.            
            if tag_tbody :        
                rows_course_infos = tag_tbody.find_all('tr')

                if len(rows_course_infos) >= 5 :
                    # 태그인 자식들만 뽑아낸다.
                    tag_children = [child for child in rows_course_infos[1].contents if child.name]

                    if len(tag_children) >= 8 :
                        course_syllabus["course_title"] = tag_children[1].text

                        course_code, section = map(str.strip, tag_children[3].text.split("/"))

                        course_syllabus["course_code"] = course_code
                        course_syllabus["section"] = section

                        course_syllabus["department"] = tag_children[5].text
                        course_syllabus["research_lab"] = tag_children[7].text

                    tag_children = [child for child in rows_course_infos[2].contents if child.name]

                    if len(tag_children) >= 8 :
                        course_syllabus["course_type"] = tag_children[1].text
                        course_syllabus["lecture_time"] = tag_children[3].text
                        course_syllabus["main_professor"] = tag_children[5].text
                        course_syllabus["joint_course"] = tag_children[7].text

                    tag_children = [child for child in rows_course_infos[3].contents if child.name]

                    if len(tag_children) >= 6 :
                        course_syllabus["lecture_room"] = tag_children[1].text
                        course_syllabus["credit_hours"] = tag_children[3].text
                        course_syllabus["email"] = tag_children[5].text

                    tag_children = [child for child in rows_course_infos[4].contents if child.name]

                    if len(tag_children) >= 8 :
                        course_syllabus["grading_type"] = tag_children[1].text
                        course_syllabus["contact_info"] = tag_children[5].text
                        course_syllabus["office_hours"] = tag_children[7].text

            syllabus_sections = div_course_syllabus.find_all('table', class_='table table-bordered')          

            if len(syllabus_sections) >= 15 :
                course_syllabus["lecture_overview"] = slead_syllabus.syllabus_lecture_overview(syllabus_sections[0])
                course_syllabus["prerequisite_courses"] = slead_syllabus.syllabus_prerequisite_courses(syllabus_sections[1])                 
                course_syllabus["lecture_style"] = slead_syllabus.syllabus_lecture_style(syllabus_sections[2])
                course_syllabus["course_objectives"] = slead_syllabus.syllabus_course_objectives(syllabus_sections[3])
                course_syllabus["hi_five_skills"] = slead_syllabus.syllabus_hi_five_skills(syllabus_sections[4])                  
                course_syllabus["competency_info"] = slead_syllabus.syllabus_competency_info(syllabus_sections[5])                  
                course_syllabus["learning_objectives"] = slead_syllabus.syllabus_learning_objectives(syllabus_sections[6])                  
                course_syllabus["grading_weight"] = slead_syllabus.syllabus_grading_weight(syllabus_sections[7])         
                course_syllabus["attendance_threshold"] = slead_syllabus.syllabus_attendance_threshold(syllabus_sections[8])         
                course_syllabus["teaching_method"] = slead_syllabus.syllabus_teaching_method(syllabus_sections[9])         
                course_syllabus["class_policy"] = slead_syllabus.syllabus_class_policy(syllabus_sections[10])         
                course_syllabus["textbooks"] = slead_syllabus.syllabus_textbooks(syllabus_sections[11])        
                # 12 -> 주차 별 수업 계획은 스킵(강의 정보에서 더 정확하게 보여줌) 
                course_syllabus["additional_notes"] = slead_syllabus.syllabus_additional_notes(syllabus_sections[13])         
                course_syllabus["accessibility_support"] = slead_syllabus.syllabus_accessibility_support(syllabus_sections[14])         

            return { "status" : "success", "data": course_syllabus }
        else:
            return { "status" : "failed", "error_msg": f"session.get fail! status code: {response.status_code}" }
    except (requests.RequestException, TypeError, KeyError) as e:
        return { "status" : "failed", "error_msg": f"exception: {e}" }

def get_course_participants(login_user : SLeadMember, course_id : str ) :
    """
    강좌의 참여자 목록을 조회한다.

    :리턴값: 성공(True,참여자 리스트), 실패(False,"에러메시지")\n

    [
        {
            "number" : "", # 번호
            "user_id_number" : "", # 유저 아이디 넘버
            "department" : "", # 학과(전공)
            "student_number" : "", # 학번
            "name" : "", # 이름
            "role" : "", # 역할
            "last_access" : "" # 최근접속
        }
    ]
    """
    session = login_user.get_session() 

    if session is None :
        return False, "none session"

    course_participants = []

    base_url = "https://smartlead.hallym.ac.kr/user/users.php"  
    url = f"{base_url}?id={course_id}"

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            tag_table = soup.find("table", {"id": "participants"})

            if tag_table is None :
                return False, "tag_table is not found"
            
            tag_tbody = tag_table.find("tbody")

            if tag_tbody is None :
                return False, "tag_tbody is not found"

            tag_rows = tag_tbody.find_all("tr")

            for tag_row in tag_rows :
                # 정보가 없는 빈행은 스킵
                if "emptyrow" in tag_row.get("class", []):
                    continue

                tag_tds = tag_row.find_all("td")

                if len(tag_tds) < 9 :
                    return False, "tag_tds len smaller than 9"
                
                participant_info = {
                    "number" : "", # 번호
                    "user_id_number" : "", # 유저 아이디 넘버
                    "department" : "", # 학과(전공)
                    "student_number" : "", # 학번
                    "name" : "", # 이름
                    "role" : "", # 역할
                    "last_access" : "" # 최근접속
                }

                participant_info["number"] =tag_tds[1].get_text(separator=" ").strip()

                a_tag = tag_tds[2].find("a")

                if a_tag :
                    result, user_id_numer = extract_query_value_from_atag(a_tag, "id")
                    if result :
                        participant_info["user_id_number"] = user_id_numer
                participant_info["department"] =tag_tds[3].get_text(separator=" ").strip()
                participant_info["student_number"] =tag_tds[4].get_text(separator=" ").strip()
                participant_info["name"] = tag_tds[5].get_text(separator=" ").strip()

                tag_span = tag_tds[6].find("span")
                if tag_span:
                    participant_info["role"] = tag_span.get_text(separator=" ").strip()

                participant_info["last_access"] = tag_tds[7].get_text(separator=" ").strip()

                course_participants.append(participant_info)

            return True, course_participants
        else:
            return False, f"session.get fail! status code: {response.status_code}"
    except (requests.RequestException, TypeError, KeyError) as e:
        return False, f"error: {e}"

def get_noncurricular_courses( login_user : SLeadMember, year : str ) :
    """
    해당 연도의 비교과 강의 목록을 조회한다.
    [
        {
            "year" : "",   # 년도
            "name" : "",   # 강좌명
            "semester" : "",   # 학기
            "professor" : "",  # 교수
        }    
    ]
    """
    session = login_user.get_session() 

    if session is None :
        return False, "none session"

    irregular_courses = []

    for page_index in range(1,21) :
        base_url = "https://smartlead.hallym.ac.kr/local/ubassistant/irregular.php"  
        url = f"{base_url}?year={year}&page={page_index}"

        try :
            response = session.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')

                tag_table = soup.find('table')

                if tag_table is None :
                    return False, "tag_table not found"
                
                tag_tbody = tag_table.find('tbody')

                if tag_tbody is None :
                    return False, "tag_tbody not found"

                tag_tr_rows = tag_tbody.find_all("tr")

                if len(tag_tr_rows) <= 0 :
                    break

                for tag_row in tag_tr_rows :
                    tag_tds = tag_row.find_all("td")

                    if len(tag_tds) >= 4 :                                       
                        courses_info = {
                            "year" : tag_tds[0].get_text(strip=True),   # 년도
                            "name" : tag_tds[1].get_text(strip=True),   # 강좌명
                            "semester" : tag_tds[2].get_text(strip=True),   # 학기
                            "professor" : tag_tds[3].get_text(strip=True),  # 교수
                        }

                        irregular_courses.append(courses_info)
            else:
                return False, f"session.get fail! status code: {response.status_code}"
        except (requests.RequestException, TypeError, KeyError) as e:
            return False, f"error: {e}"

    return True, irregular_courses

def get_course_settings( login_user : SLeadMember, course_id : str ) :
    session = login_user.get_session() 

    if session is None :
        return False, "none session"

    base_url = "https://smartlead.hallym.ac.kr/local/ubion/setting/course.php"  
    url = f"{base_url}?id={course_id}"

    course_settings = {}

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            div_local_ubion_setting = soup.find('div', class_='local_ubion_setting')

            if div_local_ubion_setting is None :
                return False, "div_local_ubion_setting not found"
   
            tag_form = div_local_ubion_setting.find('form')

            if tag_form is None :
                return False, "tag_form not found"

            return True, course_settings
        else:
            return False, f"session.get fail! status code: {response.status_code}"
    except (requests.RequestException, TypeError, KeyError) as e:
        return False, f"error: {e}"
    
def get_weekly_learning_activities( login_user : SLeadMember, course_id : str ) -> dict: 
    """
    강좌의 주차 별 학습 활동을 조회한다.
    
    Args:
        login_user (SLeadMember) :  유저 인스턴스
        course_id (str) : 강좌 아이디

    Returns:
            dict: 조회 결과 포함하는 딕셔너리
            - 성공:
                {
                    "status": True,
                    "data": [
                    {
                    "name" : "",          # 주차 이름
                    "lecture_materials" : # 자료 목록
                    [
                        {
                        "name" : "", # 자료 이름
                        "id" : "", # 자료 아이디
                        "url" : "", # 자료의 연결 url
                        "type" : "", # 자료 종류(파일/과제) : 'ubfile' : 파일(다운로드가능) 'assign' : 과제(과제 페이지)  folder(폴더 페이지)/vod(동영상 페이지)/quiz(퀴즈 페이지)/url(외부 링크)
                        }
                    ],   
                    }
                ]
            - 실패: 
                {
                    "status": False,
                    "error_msg": ""
                }
    """
    session = login_user.get_session() 

    if session is None :
        return {"status" : False,"error_msg" :"none session"}

    base_url = "https://smartlead.hallym.ac.kr/course/view.php"  
    url = f"{base_url}?id={course_id}"

    learning_activities = []

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            
            div_total_sections = soup.find('div', class_='total-sections')

            if div_total_sections is None :
                return {"status" : False, "error_msg": "div_total_sections not found"}

            ul_ubsweeks = div_total_sections.find('ul', class_='ubsweeks')

            if ul_ubsweeks is None :
                return {"status" : False,"error_msg": "ul_ubsweeks not found"}
            
            li_rows = ul_ubsweeks.find_all('li')

            for li_row in li_rows :
                div_content = li_row.find('div', class_='content')

                if div_content is None :
                    continue

                lecture_week_info = {
                    "name" : "",                # 주차 이름
                    "lecture_materials" : [],   # 자료 목록
                }

                # 주차 이름 조회        
                h3_section_name = div_content.find('h3', class_='sectionname')

                if h3_section_name :
                    span_section_name = h3_section_name.find('span')

                    if span_section_name :
                        section_name = span_section_name.get_text(strip=True)
                        lecture_week_info["name"] = section_name

                # 자료 목록 조회
                tag_ul = div_content.find('ul')

                if tag_ul :
                    li_material_rows = tag_ul.find_all('li')

                    for li_material_row in li_material_rows :
                        material_info = {
                            "name" : "", # 자료 이름
                            "id" : "", # 자료 아이디
                            "url" : "", # 자료의 연결 url
                            "type" : "", # 자료 종류(파일/과제) : 'ubfile' : 파일(다운로드가능) 'assign' : 과제(과제 페이지)  folder(폴더 페이지)/vod(동영상 페이지)/quiz(퀴즈 페이지)/url(외부 링크)                          
                        }

                        div_activityinstance = li_material_row.find('div', class_='activityinstance')

                        if div_activityinstance is None :
                            continue

                        tag_a_material = div_activityinstance.find('a')

                        if tag_a_material :
                            material_info["name"] = tag_a_material.get_text(strip=True)
                            _, material_info["id"] = extract_query_value_from_atag(tag_a_material,'id')
                            
                            material_href = tag_a_material.get("href")
                            if material_href:
                                material_info["url"] = material_href

                                if 'ubfile' in material_href:
                                    material_info["type"] = "ubfile"
                                elif 'assign' in material_href:
                                    material_info["type"] = "assign"
                                elif 'folder' in material_href:
                                    material_info["type"] = "folder"
                                elif 'vod' in material_href:
                                    material_info["type"] = "vod"
                                elif 'quiz' in material_href:
                                    material_info["type"] = "quiz"
                                elif 'url' in material_href:
                                    material_info["type"] = "url"
                                else :
                                    pass    

                        lecture_week_info["lecture_materials"].append(material_info)    

                learning_activities.append(lecture_week_info)

            return {"status" : True, "data" : learning_activities }
        else:
            return {"status" : False, "error_msg": f'session.get fail! status code: {response.status_code}'}
    except (requests.RequestException, TypeError, KeyError) as e:
        return {"status" : False, "error_msg" : f'exception -- {e}'}

def get_my_h_square_courses(login_user : SLeadMember) :
    """
    접속한 유저의 나의 H-Square 강좌를 조회한다.

    :리턴값: 성공(True,강좌 목록), 실패(False,"에러메시지")\n
    강좌 목록 리스트(리스트가 비었으면 참여 강좌 없음)
    [
        {
            "access_level" : "",    # 공개여부
            "category" : "",    # 분류(예:기타,)
            "title" : "", # H-Square 명 (제목)
            "course_date" : "", # 강좌 기간
            "role"  : "", # 역할(예:회원,운영자)
            "course_id" : "", # 강좌 아이디 
            "status" : "", # 상태
        }
    ]
    """    
    course_list = []

    session = login_user.get_session() 

    if session is None :
        return {"status" : "failed","error_msg": "none session"}

    base_url = "https://smartlead.hallym.ac.kr/local/ubeclass/my.php"  
    url = f"{base_url}"

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            div_eclass_list = soup.find('div', class_='eclass-list')

            if div_eclass_list :
                tbody_courses = div_eclass_list.find("tbody")

                if tbody_courses is None :
                    return {"status" : "failed","error_msg": "none tbody_courses"}

                tr_courses = tbody_courses.find_all('tr')

                for tr_course in tr_courses:
                    course_info = {
                                    "access_level" : "",    # 공개여부
                                    "category" : "",    # 분류(예:기타,)
                                    "title" : "", # H-Square 명 (제목)
                                    "course_date" : "", # 강좌 기간
                                    "role"  : "", # 역할(예:회원,운영자)
                                    "course_id" : "", # 강좌 아이디 
                                    "status" : "", # 상태
                                    }
                    
                    td_course_column = tr_course.find_all("td")

                    if len(td_course_column) != 7:
                        continue

                    course_info["access_level"] = td_course_column[1].get_text(strip=True)    
                    course_info["category"] = td_course_column[2].get_text(strip=True)    

                    tag_a_title = td_course_column[3].find('a')    

                    if tag_a_title :
                        course_info["title"] = tag_a_title.get_text(strip=True)
                        course_info["course_id"] = extract_query_value_from_atag(tag_a_title,"id")

                    tag_span_date = td_course_column[3].find('span', class_='coursedate')    

                    if tag_span_date :
                        course_info["course_date"] = tag_span_date.get_text(strip=True)

                    course_info["role"] = td_course_column[4].get_text(strip=True)    

                    div_badge = td_course_column[5].find('div', class_='badge')

                    if div_badge :
                        course_info["status"] = div_badge.get_text(strip=True)

                    course_list.append( course_info )                               

            return {"status":"success", "data" : course_list}
        else:
            return {"status":"failed", "error_msg" : f"page request fail!!! status code: {response.status_code}"}
    except (requests.RequestException, TypeError, KeyError) as e:
        return {"status":"failed", "error_msg" : f"error: {e}"}