from typing import List,Tuple
import requests
from .slead_login import SLeadLogin 

class SLeadMember:
    def __init__(self):
        self.login_object = None
 
    def login( self, user_id : str, user_pw : str ) -> bool :
        """
        스마트리드 사이트에 로그인한다.

        Args:
            None

        Returns:
            bool:
                - 성공:
                    True
                - 실패: 
                    False
        """
        try :
            self.login_object = SLeadLogin()

            result = self.login_object.user_login(user_id, user_pw)

            return result
        except Exception as e:  
            print(f"Login error: {e}") 
            return False

    def logout( self ) -> tuple[bool, str] :
        """
        스마트리드 사이트에서 로그인한다.

        Args:
            None

        Returns:
            tuple[bool, str]:
                - 성공:
                    True,""
                - 실패: 
                    False,"error_msg"
         """
        if self.login_object is None :
            return False, {"error_msg" : "login_object none"}

        return self.login_object.user_logout()
        
    def get_user_profile(self) -> Tuple[bool, dict]:     
        """
        접속한 유저의 개인정보 조회      

        Args:
            None

        Returns:
            tuple[bool, dict]:
                - 성공:
                    True,
                    {
                       "student_number" : "", # 학번
                       "department" : "", # 학과
                       "name_kor" : "", # 한국어 이름
                       "name_eng" : "", # 영어 이름
                    }
                - 실패: 
                    False,{"error_msg": ""}
         """

        if self.login_object is None :
            return False, {"error_msg" : "login_object none"}

        result, data_or_error = self.login_object.get_user_profile()

        if result :
            return  result, data_or_error
        else :
            return  result, {"error_msg" : data_or_error}

    def get_session(self) -> requests.Session : 
        if self.login_object is None :
            return None
        return self.login_object.get_session()

    def get_sesskey(self) -> tuple[bool, str] : 
        if self.login_object is None :
            return False, "login_object none"
        
        return self.login_object.get_sesskey()

    def get_user_moodle_id(self) -> Tuple[bool, str] : 
        if self.login_object is None :
            return False, "login_object none"
        
        return self.login_object.get_user_moodle_id()
