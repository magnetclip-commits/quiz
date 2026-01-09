import requests
from bs4 import BeautifulSoup
from typing import List,Tuple, Dict, Union
from .slead_member import SLeadMember
from .slead_parser_helper import extract_query_value_from_atag
import re

def get_active_course_notices(login_user : SLeadMember) :
    """
    현재 참여중인 강좌의 공지 목록을 조회한다.

    리턴: 성공(True,공지목록 리스트), 실패(False,"실패 메시지")
    {
        "writer" : "", : 작성자
        "title" : "", : 공지 제목
        "date" : "", : 공지 날짜
        "course_name" : "", : 강좌 이름
        "module_id" : "", : course module id
        "bwid" : "" : 게시물 번호
    }    
    """
    session = login_user.get_session() 

    if session is None :
        return False, "none session"

    base_url = "https://smartlead.hallym.ac.kr/mod/ubboard/my.php"  
    url = f"{base_url}"

    notice_list = []

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            #<div class="ubboard_container"> --> 이 요소안에 공지 목록이 존재
            div_ubboard_container = soup.find('div', class_='ubboard_container')

            if div_ubboard_container is None :
                return False, "div_ubboard_container not found"        

            #<table class="ubboard_table table">
            ubboard_table = div_ubboard_container.find('table', class_='ubboard_table')

            if ubboard_table is None:
                return True, notice_list

            notice_table_body = ubboard_table.find('tbody')

            if notice_table_body is None:
                return True, notice_list

            # 각각의 공지 정보가 <tr>안에서 수집됨
            tr_notices = notice_table_body.find_all("tr")

            for notice_row in tr_notices :
                notice_info = {
                                "writer" : "",
                                "title" : "",
                                "date" : "",
                                "course_name" : "",
                                "module_id" : "", # course module id
                                "bwid" : "" # 게시물 번호
                                }    
                                    
                notice_data = notice_row.find_all("td")

                if len(notice_data) >= 5 :
#                        notice_number = notice_data[0].text
                    notice_info["writer"] = notice_data[2].text
                    notice_info["date"] = notice_data[3].text
#                       notice_view_count = notice_data[4].text
                    notice_title_data = notice_data[1]

                    a_tag = notice_title_data.find('a')

                    if a_tag:
                        # <a href="https://smartlead.hallym.ac.kr/mod/ubboard/article.php?ls=15&amp;id=509985&amp;bwid=234497">
                        # 	 <strong>[에듀테크 test용]</strong>
                        #	  테스트공지
                        # </a>
                        # 이 링크같은 주소에서 id,bwid,에듀테크 test용 : 강좌명,테스트공지 : 공지제목 를 뽑아낸다. 
                        href = a_tag.get("href", "")

                        match = re.search(r"id=(\d+)", href)

                        if match:
                            notice_module_id = match.group(1)  
                            notice_info["module_id"] = notice_module_id

                        match = re.search(r"bwid=(\d+)", href)    

                        if match:
                            notice_bwid = match.group(1)  
                            notice_info["bwid"] = notice_bwid

                        # 강좌명                            
                        strong_tag = a_tag.find("strong")

                        if strong_tag:      
                            strong_text = strong_tag.text.strip()

                            match = re.search(r"\[(.*?)\]", strong_text)

                            if match:
                                course_name = match.group(1)     

                                notice_info["course_name"] = course_name
                        else :
                            strong_text = ""

                        a_tag_text = a_tag.text
                        notice_title = a_tag_text.replace(strong_text, "").strip()

                        notice_info["title"] = notice_title

                    notice_list.append(notice_info)                                

            return True, notice_list
        else:
            return False, f"session.get fail! status code: {response.status_code}"
    except (requests.RequestException) as e:
        return False, f"error: {e}"
        
def get_post_board_notices( login_user : SLeadMember, course_id : str, icon_img_name : str, page : int = 1 ) :
    """
    게시판 목록 조회한다.(게시판을 조회하는 것은 동일하나 강의개요에서 게시판 링크 제목만 다른경우 사용한다. )

    Args:
        login_user: 로그인된 사용자 세션
        course_id: 강좌 ID
        icon_img_name: 게시판 아이콘 이미지 이름
        page: 조회할 페이지 번호 (기본값: 1)

    {
        "board_module_id" : "", # 게시판 모듈 아이디
        "notice_list" : # 게시글 목록
        [
            {
                "number" : "", # 글 번호
                "title" : "", # 글 제목
                "post_url" : "", # 게시글 링크
                "writer" : "", # 글 작성자
                "write_date" : "", # 글 작성일
            }
        ]
    }   
    """
    session = login_user.get_session() 

    if session is None :
        return False, "none session"

    base_url = "https://smartlead.hallym.ac.kr/course/view.php"  
    url = f"{base_url}?id={course_id}"

    course_notices = {
        "board_module_id" : "", # 공지 게시판 모듈 아이디
        "notice_list" : []
    }

    try :
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            
            div_course_box_top = soup.find('div', class_='course-box-top')

            if div_course_box_top is None :
                return False, f"div_course_box_top not found"

            ul_weeks = div_course_box_top.find('ul', class_='weeks')

            if ul_weeks is None :
                return False, f"ul_weeks not found"

            li_section_0 = ul_weeks.find('li', {'id': 'section-0'})

            if li_section_0 is None :
                return False, f"li_section_0 not found"

            div_content = li_section_0.find('div', class_='content')

            if div_content is None :
                return False, f"div_content not found"

            ul_section = div_content.find('ul', class_='section')

            if ul_section is None :
                return False, f"ul_section not found"

            li_ubboards = ul_section.find_all('li', class_='ubboard')

            li_notice_board = None

            for li_ubboard in li_ubboards :
                tag_a_li_ubboard = li_ubboard.find('a')

                """
                강의개요에 있는 게시판 링크를 구별하는 방법이 없어(이름도 변경가능한듯하여)
                아이콘 이미지로 원하는 게시판을 구별한다.
                """
                if tag_a_li_ubboard :
                    tag_img = tag_a_li_ubboard.find('img')

                    if tag_img :
                        # img 링크 경로 안에 원하는 아이콘 이미지 이름이 있는지 판별
                        if icon_img_name in tag_img['src'].lower():
                            li_notice_board = li_ubboard 
                            break

            if li_notice_board is None:                
                return False, f"li_notice_board not found"
            
            # 실제 과목공지 게시판 공지 링크를 뽑아낸다.
            tag_a_baord = li_notice_board.find('a')

            if tag_a_baord is None :
                return False, f"tag_a_baord not found"

            board_url = tag_a_baord.get("href","")

            if board_url == "" :
                 return False, "href not found" 
            
            result, board_module_id = extract_query_value_from_atag( tag_a_baord, "id" )

            if not result :
                 return False, f"extract_query_value_from_atag error{board_url}" 
        else:
            return False, f"session.get fail! status code: {response.status_code}"      
        
        # 과목공지 게시판에 접속 - 페이지 파라미터 추가
        if page > 1:
            # 페이지 파라미터 추가 (page=N)
            if '?' in board_url:
                url = f"{board_url}&page={page}"
            else:
                url = f"{board_url}?page={page}"
        else:
            url = board_url
        
        # 과목공지 게시판에 접속
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            table_ubboard_table = soup.find('table', class_='ubboard_table')

            if table_ubboard_table is None :
                 return False, "table_ubboard_table not found" 
            
            tbody_ubboard_table = table_ubboard_table.find('tbody')

            if tbody_ubboard_table is None :
                 return False, "tbody_ubboard_table not found" 
            
            tr_rows = tbody_ubboard_table.find_all('tr')

            course_notices["board_module_id"] = board_module_id

            for tr_row in tr_rows :
                td_column_datas = tr_row.find_all('td')

                len_td_column_datas = len(td_column_datas)

                if len_td_column_datas != 6 and len_td_column_datas != 5 :
                     if len_td_column_datas == 1:
                         # 공지된 게시글이 없음
                         return True, course_notices   

                     return False, f"invalid td count{len(td_column_datas)}" 
                
                start_column_index = 0

                # 교수자는 게시판 첫번째 항목이 선택박스임
                if td_column_datas[0].find('input', {'type': 'checkbox'}) is not None :
                    start_column_index = 1

                post_title = ""
                post_url = ""

                tag_a_title = td_column_datas[start_column_index+1].find('a')

                if tag_a_title:
                    post_title = tag_a_title.get_text(strip=True)
                    post_url = tag_a_title.get("href","")

                notice_info = {
                    "number" : td_column_datas[start_column_index].get_text(strip=True), # 글 번호
                    "title" : post_title, # 글 제목
                    "post_url" : post_url, # 게시글 링크
                    "writer" : td_column_datas[start_column_index+2].get_text(strip=True), # 글 작성자
                    "write_date" : td_column_datas[start_column_index+3].get_text(strip=True), # 글 작성일
                }

                course_notices["notice_list"].append(notice_info)
            
            return True, course_notices
        else:
            return False, f"session.get fail! status code: {response.status_code}"      
        
    except (requests.RequestException, TypeError, KeyError) as e:
        return False, f"error: {e}"

def get_course_notices( login_user : SLeadMember, course_id : str, page : int = 1 ) :
    """
    강의의 과목공지 게시판 목록을 조회한다.

    Args:
        login_user: 로그인된 사용자 세션
        course_id: 강좌 ID
        page: 조회할 페이지 번호 (기본값: 1)

    {
        "board_module_id" : "", # 게시판 모듈 아이디
        "notice_list" : # 게시글 목록
        [
            {
                "number" : "", # 글 번호
                "title" : "", # 글 제목
                "post_url" : "", # 게시글 링크
                "writer" : "", # 글 작성자
                "write_date" : "", # 글 작성일
            }
        ]
    }   
    """
    return get_post_board_notices(login_user, course_id, "ubboard_notice", page)
        
def get_course_qna( login_user : SLeadMember, course_id : str, page : int = 1 ) :
    """
    강의의 FAQ 질문/질의응답 게시판 목록을 조회한다.

    Args:
        login_user: 로그인된 사용자 세션
        course_id: 강좌 ID
        page: 조회할 페이지 번호 (기본값: 1)

    {
        "board_module_id" : "", # 게시판 모듈 아이디
        "notice_list" : # 게시글 목록
        [
            {
                "number" : "", # 글 번호
                "title" : "", # 글 제목
                "post_url" : "", # 게시글 링크
                "writer" : "", # 글 작성자
                "write_date" : "", # 글 작성일
            }
        ]
    }   
    """
    return get_post_board_notices(login_user, course_id, "ubboard_qna", page)

def get_notice_content( login_user : SLeadMember, post_url : str ) :
    """
    게시글 내용을 조회한다.

    Args:
        login_user: 로그인된 사용자 세션
        post_url: 게시글 URL

    Returns:
        성공(True, 게시글 데이터), 실패(False, 에러 메시지)
        {
            "title": "", # 게시글 제목
            "writer": "", # 작성자
            "date": "", # 작성일
            "content": "", # 게시글 내용 (HTML 포함)
            "html": "", # 전체 HTML (디버깅용)
            "debug_info": {} # 디버깅 정보
        }
    """
    session = login_user.get_session() 

    if session is None :
        return False, "none session"
    
    try :
        response = session.get(post_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            
            debug_info = {
                "classes_found": [],
                "ids_found": []
            }
            
            # 페이지에 있는 주요 클래스와 ID 수집 (디버깅용)
            for element in soup.find_all(class_=True):
                for cls in element.get('class'):
                    if cls not in debug_info["classes_found"]:
                        debug_info["classes_found"].append(cls)
            
            for element in soup.find_all(id=True):
                if element.get('id') not in debug_info["ids_found"]:
                    debug_info["ids_found"].append(element.get('id'))
            
            # 게시글 제목 찾기 - 'subject' 클래스 (디버깅 정보에서 확인됨)
            title_element = soup.find('div', class_='subject')
            title = title_element.get_text(strip=True) if title_element else ""
            
            # 작성자 정보 찾기 - 'writer' 클래스 (디버깅 정보에서 확인됨)
            writer_element = soup.find('div', class_='writer')
            writer = writer_element.get_text(strip=True) if writer_element else ""
            
            # 작성일 찾기 - 'date' 클래스 (디버깅 정보에서 확인됨)
            date_element = soup.find('div', class_='date')
            date = date_element.get_text(strip=True) if date_element else ""
            
            # 게시글 내용 찾기 - 'text_to_html' 클래스 (디버깅 정보에서 확인됨)
            content_element = soup.find('div', class_='text_to_html')
            
            # 내용이 없을 경우 다른 클래스 시도
            if not content_element:
                content_element = soup.find('div', class_='content')
                
            # 여전히 없을 경우 div.well-sm 안의 내용 찾기
            if not content_element:
                well_sm = soup.find('div', class_='well-sm')
                if well_sm:
                    # well-sm 내부의 첫 번째 div가 내용을 담고 있을 가능성이 높음
                    content_element = well_sm.find('div')
            
            content = str(content_element) if content_element else ""
            
            # 게시글 내용 텍스트 추출
            content_text = ""
            if content_element:
                content_text = content_element.get_text(strip=True)
            
            notice_data = {
                "title": title,
                "writer": writer,
                "date": date,
                "content": content,
                "content_text": content_text,  # 텍스트만 추출한 내용 추가
                "html": str(soup),  # 전체 HTML 저장 (디버깅용)
                "debug_info": debug_info
            }
            
            return True, notice_data
        else:
            return False, f"session.get fail! status code: {response.status_code}"
    
    except (requests.RequestException, TypeError, KeyError) as e:
        return False, f"error: {e}"
