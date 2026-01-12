import argparse
import requests
import json
import sys
import time

# 기본 설정
DEFAULT_API_URL = "http://localhost:8085"

def sync_lms(user_id, cls_id, year, semester, api_base_url):
    print(f"=== LMS 파일 동기화 시작 ===")
    print(f"User: {user_id}, Class: {cls_id}, Year: {year}, Semester: {semester}")
    
    # 1. LMS 파일 목록 갱신 (Crawling)
    print("\n[1단계] LMS 파일 목록 갱신 중...")
    url_list = f"{api_base_url}/file/lmsfilelist"
    payload_list = {
        "user_id": user_id,
        "cls_id": cls_id,
        "year": year,
        "semester": semester
    }
    
    try:
        resp = requests.post(url_list, json=payload_list)
        if resp.status_code == 200:
            print(" -> 성공: 파일 목록이 갱신되었습니다.")
        else:
            print(f" -> 실패: Status {resp.status_code}, {resp.text}")
            return False, 0
    except Exception as e:
        print(f" -> 에러: {e}")
        return False, 0

    # 2. 다운로드 대상 조회
    print("\n[2단계] 미처리 파일(다운로드 대상) 조회 중...")
    url_target = f"{api_base_url}/file/materiallist"
    # materiallist API는 user_id 없이 cls_id, year, semester만 받음 (routers/file.py 참고)
    payload_target = {
        "cls_id": cls_id,
        "year": year,
        "semester": semester
    }
    
    target_files_m = []
    target_files_v = []
    
    try:
        resp = requests.post(url_target, json=payload_target)
        if resp.status_code == 200:
            files = resp.json()
            print(f" -> 총 {len(files)}개의 대상 파일이 발견되었습니다.")
            
            for f in files:
                m_type = f.get('material_type')
                f_id = f.get('file_id')
                f_name = f.get('material_nm')
                
                if m_type == 'vod':
                    target_files_v.append(f_id)
                else:
                    target_files_m.append(f_id)
                    
            print(f"    - 영상(VOD): {len(target_files_v)}개")
            print(f"    - 자료(PDF/etc): {len(target_files_m)}개")
            
            if not target_files_m and not target_files_v:
                print(" -> 처리할 파일이 없습니다. 동기화를 종료합니다.")
                return True, 0 # 성공, 0개 처리

        else:
            print(f" -> 실패: Status {resp.status_code}, {resp.text}")
            return False, 0
    except Exception as e:
        print(f" -> 에러: {e}")
        return False, 0

    # 3. 작업 트리거 (다운로드 및 임베딩 요청)
    print("\n[3단계] 다운로드 및 임베딩 작업 요청 중...")
    url_download = f"{api_base_url}/file/materials/download"
    payload_download = {
        "user_id": user_id,
        "cls_id": cls_id,
        "file_ids_m": target_files_m,
        "file_ids_v": target_files_v
    }
    
    try:
        resp = requests.post(url_download, json=payload_download)
        if resp.status_code == 200:
            print(f" -> 성공: {resp.json().get('message')}")
            print("\n모든 작업이 큐에 등록되었습니다.")
            print("진행 상황은 Celery 로그 또는 웹소켓 상태를 확인해주세요.")
        else:
            print(f" -> 실패: Status {resp.status_code}, {resp.text}")
            return False, 0
    except Exception as e:
        print(f" -> 에러: {e}")
        return False, 0

    total_count = len(target_files_m) + len(target_files_v)
    return True, total_count

def main():
    parser = argparse.ArgumentParser(description="LMS 파일 전체 동기화 스크립트")
    parser.add_argument("--user", required=True, help="사용자 ID (DB에 비밀번호가 저장되어 있어야 함)")
    parser.add_argument("--cls_id", required=True, help="강의실 ID")
    parser.add_argument("--year", default="2025", help="년도 (기본: 2025)")
    parser.add_argument("--semester", default="20", help="학기 (기본: 20)")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API Base URL (기본: {DEFAULT_API_URL})")
    
    args = parser.parse_args()
    
    sync_lms(args.user, args.cls_id, args.year, args.semester, args.url)

if __name__ == "__main__":
    main()
