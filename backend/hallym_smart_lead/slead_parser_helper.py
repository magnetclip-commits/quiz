import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

def extract_text_child_a_tag( parent_tag ) :
    """
    부모 요소의 자식 <a> 태그에서 텍스트를 추출한다.
    예)
    <td "><a href="https://smartlead.hallym.ac.kr/mod/ubboard/view.php?id=482167">게시판: 과목공지</a></td>
    이 코드에서 '게시판: 과목공지' 를 추출
    """
    extract_text = ""

    tag_a = parent_tag.find('a')

    if tag_a :
        extract_text = tag_a.get_text(strip=True)

    return extract_text    

def element_has_class( tag, class_name ):
    """
    태그가 특정 클래스를 포함하는지 확인하는 함수
    """
    return tag.has_attr('class') and class_name in tag.get('class', [])

def extract_query_value_from_atag( a_tag : str, key : str ):
    '''
    a 태그의 href의 쿼리에서 입력된 키 값에 해당하는 값을 리턴한다.
    예)
    <a href="https://smartlead.hallym.ac.kr/course/view.php?id=18555" class="coursename" target="_blank">2024학년도 </a>

    extract_query_value_from_atag(a_tag,"id") 를 호출하면 18555를 리턴한다.
    '''
    if not a_tag :
        return False, "" 

    href = a_tag.get("href","")

    if href == "" :
        return False, "" 

    parsed_url = urlparse(href)
    query_params = parse_qs(parsed_url.query)

    extract_value = query_params.get(key, [None])[0]

    if extract_value is None :
        return False, ""
    
    return True, extract_value

def extract_learning_progress_table_headers(tag_thead):
    """ 
    학습 현황 테이블 헤더(<thead>)의 멀티 레벨 헤더 구조를 colspan, rowspan을 계산하여 2차원 배열로 리턴한다.

    [
        [번호,이름,학번,강의개요,강의개요,강의개요]      
        [번호,이름,학번,과목공지,과목공지,과목공지]      
        [번호,이름,학번,보기,쓰기,댓글]
    ]          
    """

    header_rows = tag_thead.find_all("tr")  

    # 테이블에서 최대 컬럼 개수
    max_columns = 0  
    # 테이블에서 최대 행 개수
    max_rows = 0

    header_matrix = []  # 헤더 정보를 저장할 리스트
    row_index = 0

    # 헤더 행을 분석하여 2D 리스트로 구성**
    for row in header_rows:
        headers = []
        col_index = 0  # 현재 열 인덱스 추적

        for th in row.find_all("th"):
            colspan = int(th.get("colspan", 1))  # colspan이 없으면 기본값 1
            rowspan = int(th.get("rowspan", 1))  # rowspan이 없으면 기본값 1

            text = th.get_text(strip=True)

            if text == "" :
                img_tag = th.find('img')

                if img_tag:
                    text = img_tag.get("title", "")

            # colspan만큼 동일한 헤더 확장
            for _ in range(colspan):
                headers.append({"text": text, "rowspan": rowspan, "col": col_index})
                col_index += 1

            if row_index + rowspan > max_rows :    
                max_rows = row_index + rowspan

        max_columns = max(max_columns, col_index)  # 최대 컬럼 개수 업데이트
        header_matrix.append(headers)  # 헤더 행 추가

        row_index += 1

    # rowspan 고려하여 아래 행에 헤더 상속**
    final_headers = [[""] * max_columns for _ in range(max_rows)]

    row_index = 0
    col_index = [0] * max_rows

    for headers in header_matrix:
        cur_col_index = col_index[row_index]

        for header in headers:
            text = header["text"]
            col = header["col"]
            rowspan = header["rowspan"]

            # 헤더 텍스트 채우기
            final_headers[row_index][cur_col_index] = text

            # rowspan이 있으면 다음 행에도 유지
            if rowspan > 1:
                for i in range(1, rowspan):
                    final_headers[row_index+i][cur_col_index] = text
                    col_index[row_index+i] += 1

            cur_col_index += 1        

        row_index += 1                

    #소스에서 마지막 행(보기,쓰기,댓글.......)은 <tr> 태그가 누락되어 따로 처리해 줘야 한다. 
    th_tags = tag_thead.find_all('th',recursive=False)

    cur_col_index = col_index[row_index]

    for th_tag in th_tags :
        text = th_tag.get_text(strip=True)

        final_headers[row_index][cur_col_index] = text
        cur_col_index += 1        

    return final_headers, max_columns

def split_by_last_parentheses( input_text : str ) -> tuple[str,str]:
    """
    '엘렉시_교수자 (ai_edu_pr)' 와 같이 문장의 맨끝의 괄호안의 내용과 괄호 이전 내용을
     추출하는 패턴일 경우 이 함수를 사용한다.('엘렉시_교수자' 와 'ai_edu_pr' 을 추출)
    """

    front_text = "" 
    back_text = ""

    # 뒤쪽 가로안의 내용을 추출
    match = re.search(r"\(([^()]*)\)$", input_text)               

    if match :
        back_text = match.group(1)

        # 마지막 괄호 부분을 포함해 제거한 앞쪽 문자열만 추출
        front_text = re.sub(r"\s*\([^()]*\)$", "", input_text)  # 마지막 괄호 및 앞 공백 제거
        front_text.strip()
    else :
        # 뒤쪽 괄호가 없을 경우 원문에서 앞뒤 공백문자 제거하고 리턴    
        front_text = input_text.strip()

    return front_text, back_text     
