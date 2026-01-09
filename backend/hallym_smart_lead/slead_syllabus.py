import re
from bs4 import BeautifulSoup

def syllabus_lecture_overview( section ) :
    """
    강의 계획서 교과목개요 항목의 정보를 뽑아낸다. 
    """
    lecture_overview = "" # 수업개요

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 2 :
        td_tag = tr_rows[1].find('td')
        if td_tag:
            lecture_overview = td_tag.get_text(separator=" ").strip()
            lecture_overview = lecture_overview.replace("\r", "").replace("\n", " ")

    return lecture_overview

def syllabus_lecture_style( section ) :
    """
    강의 계획서 수업유형 정보를 뽑아낸다. 

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
    """
    lecture_style = {
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

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 4 :
        td_tags = tr_rows[3].find_all('td')
        if len(td_tags) >= 9:
            lecture_style["lecture_based"] = td_tags[0].text
            lecture_style["discussion_based"] = td_tags[1].text
            lecture_style["pbl"] = td_tags[2].text
            lecture_style["team_based"] = td_tags[3].text
            lecture_style["collaborative_individual"] = td_tags[4].text
            lecture_style["lab_practice"] = td_tags[5].text
            lecture_style["field_study"] = td_tags[6].text
            lecture_style["capstone_design"] = td_tags[7].text
            lecture_style["other"] = td_tags[8].text

    return lecture_style


def syllabus_prerequisite_courses( section ) :
    """
    강의 계획서 선수과목 정보를 뽑아낸다. 
    """
    prerequisite_courses = "" # 선수과목

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 2 :
        td_tag = tr_rows[1].find('td')
        if td_tag:
            prerequisite_courses = td_tag.text

    return prerequisite_courses

def syllabus_course_objectives(section) :
    """
    강의 계획서 교과목표를 뽑아낸다. 
    """
    course_objectives = ""

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 3 :
        td_tag = tr_rows[1].find('td')
        if td_tag:
            course_objectives = td_tag.get_text(separator=" ").strip()
            course_objectives = course_objectives.replace("\r", "").replace("\n", " ")

    return course_objectives

def syllabus_hi_five_skills(section) :
    """
    강의 계획서에서 HI FIVE전공능력을 뽑아낸다.
    """
    hi_five_skills = {
        "function" : "", # 직무 - 기업가형
        "innovation" : "", # 도전 - 혁신가형
        "various_skills" : "", # 기술 - 전문가형
        "exploration" : "", # 탐구 - 연구자원
    }

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 3 :
        th_tags = tr_rows[1].find_all('th')
        if len(th_tags) >= 4:
           # Function( 60 %) -> 이런 형식의 텍스트에서 60 % 추출
           match = re.search(r"(\d*\s?%)", th_tags[0].text)
           hi_five_skills["function"] = match.group() if match else None

           match = re.search(r"(\d*\s?%)", th_tags[1].text)
           hi_five_skills["innovation"] = match.group() if match else None

           match = re.search(r"(\d*\s?%)", th_tags[2].text)
           hi_five_skills["various_skills"] = match.group() if match else None

           match = re.search(r"(\d*\s?%)", th_tags[3].text)
           hi_five_skills["exploration"] = match.group() if match else None

    return hi_five_skills

def syllabus_competency_info(section) :
    """
    강의 계획서에서 역량정보를 뽑아낸다.
    """
    competency_info = {
        "core_competencies" : "", #핵심역량
        "sub_competencies" : "", #하위역량
    }

    tr_rows = section.find_all('tr')

    if len(tr_rows) >= 3 :
        tag_tds = tr_rows[2].find_all('td')

        competency_info["core_competencies"] = tag_tds[0].text
        competency_info["sub_competencies"] = tag_tds[1].text

    return competency_info

def syllabus_learning_objectives(section) :
    """
    강의 계획서에서 학습목표및 평가지표를 뽑아낸다.
    """
    tag_tbody = section.find('tbody')

    learning_objectives = {
        "knowledge" : [],
        "skill" : [],
        "attitude" : [],
    }

    if tag_tbody is None:
        return learning_objectives


    tr_rows = tag_tbody.find_all('tr',recursive=False)

    category = "knowledge"

    for tr_row in tr_rows :
        tag_tds = tr_row.find_all('td',recursive=False)

        if len(tag_tds) < 2 :
            continue

        objective_index = 0    

        # 첫데이터가 영역 정보 나타내는지 체크하고 건너뛴다.    
        if tag_tds[0].has_attr("rowspan") :
            tag_td_text = tag_tds[0].get_text(strip=True)
            if tag_td_text == "지식" :
                category = "knowledge" 
            elif tag_td_text == "기술" :
                category = "skill" 
            elif tag_td_text == "태도" :
                category = "attitude" 

            objective_index = objective_index +1    

        evaluation_criteria_index = objective_index +1

        objective = tag_tds[objective_index].get_text(strip=True)
        evaluation_criteria = tag_tds[evaluation_criteria_index].get_text(strip=True)

        if objective == "" :
            continue

        learning_assessment_map = {
            "lecture_objective" : "", # 학습목표
            "evaluation_criteria" : "", # 평가항목 및 평가지표
        }

        #\r,\n 제거
        objective = objective.replace("\r", "").replace("\n", " ")
        evaluation_criteria = evaluation_criteria.replace("\r", "").replace("\n", " ")

        learning_assessment_map["lecture_objective"] = objective
        learning_assessment_map["evaluation_criteria"] = evaluation_criteria

        learning_objectives[category].append(learning_assessment_map)

    return learning_objectives

def syllabus_grading_weight(section) :
    """
    강의 계획서에서 성적반영 비율을 뽑아낸다.
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
    }

    """
    grading_weight = {
        "total": "",          # 총합 
        "attendance_score": "",  # 출석 점수
        "assignment_score": "",   # 과제 점수
        "midterm_score": "",   # 중간고사 점수
        "final_score": "",   # 기말고사 점수
        "other": {  # 기타              
            "score" : "", # 기타 점수
            "description" : "", #상세내용
        }               
    }

    tr_rows = section.find_all('tr')

    if len(tr_rows) < 4 :
        return grading_weight

    tag_tds = tr_rows[3].find_all('td',recursive=False)

    if len(tag_tds) >= 7 :
        grading_weight["total"] = tag_tds[0].get_text(strip=True)
        grading_weight["attendance_score"] = tag_tds[1].get_text(strip=True)
        grading_weight["assignment_score"] = tag_tds[2].get_text(strip=True)
        grading_weight["midterm_score"] = tag_tds[3].get_text(strip=True)
        grading_weight["final_score"] = tag_tds[4].get_text(strip=True)
        grading_weight["other"]["score"] = tag_tds[5].get_text(strip=True)
        grading_weight["other"]["description"] = tag_tds[6].get_text(strip=True)

    return grading_weight

def syllabus_attendance_threshold(section) :
    """
    강의 계획서에서 출석 미달 기준 사항을 뽑아낸다.
    """
    tr_rows = section.find_all('tr')

    attendance_threshold = ""

    if len(tr_rows) >=2 :
        tag_td =  tr_rows[1].find('td')
        if tag_td :
            attendance_threshold = tag_td.get_text(strip=True)

    return attendance_threshold

def syllabus_teaching_method(section) :
    """
    강의 계획서에서 수업 운영방법을 뽑아낸다.
    """
    tr_row = section.find('tr')

    teaching_method = ""

    if tr_row is None :
        return teaching_method

    td_tag = tr_row.find('td')
    if td_tag:
        teaching_method = td_tag.get_text(separator=" ").strip()
        teaching_method = teaching_method.replace("\r", "").replace("\n", " ")

    return teaching_method

def syllabus_class_policy(section) :
    """
    강의 계획서에서 수업규정 정보 추출
    """
    tr_row = section.find('tr')

    class_policy = ""

    if tr_row is None :
        return class_policy

    td_tag = tr_row.find('td')
    if td_tag:
        class_policy = td_tag.get_text(separator=" ").strip()
        class_policy = class_policy.replace("\r", "").replace("\n", " ")

    return class_policy

def syllabus_textbooks(section) :
    """
    강의 계획서에서 교재 및 참고도서 정보 추출.
    """
    tr_rows = section.find_all('tr')

    textbooks = {
        "primary_textbook" : "",    # 주교재
        "secondary_textbook" : "",  # # 부교재 및 참고도서
    }

    if len(tr_rows) >= 3 :
        td_primary = tr_rows[1].find('td')

        if td_primary :
            textbooks["primary_textbook"] = td_primary.get_text(strip=True)      

        td_secondary = tr_rows[2].find('td')

        if td_primary :
            textbooks["secondary_textbook"] = td_secondary.get_text(strip=True)     


    return textbooks          

def syllabus_additional_notes(section) :           
    """
    강의 계획서에서 기타사항을 뽑아낸다.
    """
    additional_notes = ""

    tr_row = section.find('tr')

    if tr_row:
        td_tag = tr_row.find('td')

        if td_tag:
            additional_notes = td_tag.get_text(separator=" ").strip()
            additional_notes = additional_notes.replace("\r", "").replace("\n", " ")

    return additional_notes

def syllabus_accessibility_support(section) :
    """
    강의 계획서에서 장애학생을 위한 학습 및 평가지원 사항을 뽑아낸다.
    """
    accessibility_support = ""

    tr_row = section.find('tr')

    if tr_row:
        td_tag = tr_row.find('td')

        if td_tag:
            accessibility_support = td_tag.get_text(separator=" ").strip()
            accessibility_support = accessibility_support.replace("\r", "").replace("\t", "").replace("\n", " ")

    return accessibility_support
