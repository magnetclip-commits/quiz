import requests
from bs4 import BeautifulSoup
from typing import List,Tuple

class SLeadLogin :
    def __init__(self):
        self.session = None
        self.user_id_number = ""
        self.sess_key = ""

    def reset_logout_variabls(self) :    
        """
        유저 로그 아웃 후 필요한 변수를 초기화한다.
        """    
        self.session = None
        self.user_id_number = ""
        self.sess_key = ""

    # 사이트에 로그인할때 사용하는 계정은 user_name으로 스마트리트 사이트 내부적으로
    # 사용하는 숫자로 표시된 사용자 아이디는 user_id_number로 한다.
    def user_login( self, user_name : str, user_pw : str ) -> bool :
        """
        스마트리드 사이트에 유저 계정 정보로 로그인 수행
        """
        # 세션 생성
        self.session = requests.Session()

        # 헤더 추가
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Referer": "https://smartlead.hallym.ac.kr/login/index.php"
        }

        # 로그인에 필요한 데이터
        login_data = {
            "username": user_name,  # 사용자 아이디
            "password": user_pw,  # 사용자 비밀번호
            "logintype": "manual",
            "type": "popup_login"
        }

        # 로그인 요청 URL
        response = self.session.post("https://smartlead.hallym.ac.kr/login/index.php", data=login_data, headers=headers)
        
        # 대시보드 페이지 연결
        dashboard_response = self.session.get("https://smartlead.hallym.ac.kr/dashboard.php", headers=headers)

        soup = BeautifulSoup(dashboard_response.content, 'lxml')

        # 로그인 성공 여부 확인
        if not soup.find("body", {"id": "page-my-courses-dashboard"}):
            return False

        response, return_id_number = self.get_user_moodle_id()

        if not response :
            return False

        response, return_sess_key = self.get_sesskey()

        if not response :
            return False

        self.user_id_number = return_id_number
        self.sess_key = return_sess_key

        return True

    def user_logout( self ) -> tuple[bool, str] :
        if self.session is None :
            return False, "session is None"
        
        result, sess_key = self.get_sesskey()

        if not result:
            return False, "get_sesskey error"

        base_url = "https://smartlead.hallym.ac.kr/login/logout.php"  
        url = f"{base_url}?sesskey={sess_key}"

        try :
            response = self.session.get(url)

            if response.status_code == 200:
                self.reset_logout_variabls()
                return True, ""        
            else:
                return False, f"session.get fail! status code: {response.status_code}"
        except (requests.RequestException) as e:
            return False, f"error: {e}"        

    def close_session(self) :
        if self.session :
            self.session.close()
            self.session = None

    def get_user_moodle_id(self) -> Tuple[bool, str] : 
        """
        접속한 유저의 무들 계정 아이디 조회
        """
        if self.session is None :
            return False, "session is None"
        if self.user_id_number :
            return True, self.user_id_number

        url = "https://smartlead.hallym.ac.kr/user/user_edit.php"  

        try :
            response = self.session.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # 'id' 필드 값 추출
                # <input name="id" type="hidden" value="122333" />
                id_value = soup.find('input', {'name': 'id'})['value']
                
                return True, id_value
            else:
                return False, f"페이지 요청 실패! 상태 코드: {response.status_code}"
        except (requests.RequestException, TypeError, KeyError) as e:
            return False, f"error: {e}"
        
    def get_sesskey(self) -> Tuple[bool, str] : 
        """
        스마트리드 사이트의 세션키를 조회한다.
        """
        if self.session is None :
            return False, "session is None"

        if self.sess_key :
            return True, self.sess_key

        url = "https://smartlead.hallym.ac.kr/user/user_edit.php"  

        try :
            response = self.session.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # 'sesskey' 필드 값 추출
                # <input name="sesskey" type="hidden" value="K2EjYCSrwO" />
                sesskey_value = soup.find('input', {'name': 'sesskey'})['value']
                
                return True, sesskey_value
            else:
                return False, f"페이지 요청 실패! 상태 코드: {response.status_code}"
        except (requests.RequestException, TypeError, KeyError) as e:
            return False, f"error: {e}"
       
    def get_session(self) -> requests.Session : 
        return self.session

    def get_user_profile(self) : 
        """
        접속한 유저의 개인정보 조회
        
        {
            "student_number" : "", # 학번
            "department" : "", # 학과
            "name_kor" : "", # 한국어 이름
            "name_eng" : "", # 영어 이름
        }

        """

        user_profile = {
            "student_number" : "", # 학번
            "department" : "", # 학과
            "name_kor" : "", # 한국어 이름
            "name_eng" : "", # 영어 이름
        }

        if self.session is None :
            return False, "session is None" 

        url = "https://smartlead.hallym.ac.kr/user/user_edit.php"  

        try :
            response = self.session.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                div_user_edit = soup.find('div', class_='user_edit')

                if div_user_edit is None :
                    return False, "div_user_edit not found"

                # 학번                
                div_fitem_id_idnumber = div_user_edit.find("div", {"id": "fitem_id_idnumber"})

                # 학번이 없는 계정도 있음
                if div_fitem_id_idnumber :
                    div_form_control_static = div_fitem_id_idnumber.find('div', class_='form-control-static')

                    if div_form_control_static is None :
                        return False, "div_form_control_static not found"                

                    user_profile["student_number"] = div_form_control_static.get_text(strip=True)

                # 학과                
                div_fitem_id_department = div_user_edit.find("div", {"id": "fitem_id_department"})

                # 학과가 없는 계정도 있음
                if div_fitem_id_department :              
                    div_form_control_static = div_fitem_id_department.find('div', class_='form-control-static')

                    if div_form_control_static is None :
                        return False, "div_form_control_static not found"                

                    user_profile["department"] = div_form_control_static.get_text(strip=True)                

                # 한국어 이름                
                div_fitem = div_user_edit.find("div", {"id": "fitem_id_firstname"})

                if div_fitem is None :
                    return False, "div_fitem not found"
                
                tag_input = div_fitem.find('input')

                if tag_input is None :
                    return False, "tag_input not found"                

                user_profile["name_kor"] = tag_input.get('value','').strip()                

                # 영어 이름                
                div_fitem = div_user_edit.find("div", {"id": "fitem_id_lastname"})

                if div_fitem is None :
                    return False, "div_fitem not found"
                
                tag_input = div_fitem.find('input')

                if tag_input is None :
                    return False, "tag_input not found"                

                user_profile["name_eng"] = tag_input.get('value','').strip()                

                return True, user_profile
            else:
                return False, f"페이지 요청 실패! 상태 코드: {response.status_code}"
        except (requests.RequestException, TypeError, KeyError) as e:
            return False, f"error: {e}"
        
