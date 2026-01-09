import os
import requests
import json
import argparse
import sys
import re
import unicodedata

# 기본 설정 (커맨드라인 인자로 덮어쓸 수 있음)
DEFAULT_API_URL = "http://localhost:8085/file/upload/external"
DEFAULT_USER_ID = "admin_reindex"

def process_file(file_path, cls_id, api_url, user_id, forced_week_num=None):
    """단일 파일에 대해 API 호출"""
    try:
        # 파일 정보 추출
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lstrip('.') if ext else "bin"
        
        video_exts = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        # 파일 경로에서 정보 추출 (week_num, notices)
        if forced_week_num is not None:
             week_num = forced_week_num
        else:
             week_num = 1 # 기본값
        
        # 경로에 'notices'가 포함되어 있으면 식별
        if 'notices' in file_path.split('/'):
             file_type_cd = "N"
             week_num = 1 # 공지사항은 주차 무관 (또는 1)
        else:
            file_type_cd = "V" if ext.lower() in video_exts else "M"
            
            # 경로/파일명에서 week_N 또는 weekN 추출
            # 예: .../week_14/... -> 14, week1_file.pdf -> 1
            match = re.search(r'week_?(\d+)', file_path, re.IGNORECASE)
            if match:
                week_num = int(match.group(1))
        
        # 강제 지정이 없고, 자동 감지도 실패했으나 디렉토리명이 'week_N'인 경우 등 보완 가능
        # 현재는 forced_week_num이 있으면 위 로직을 건너뛰지는 않지만 덮어씌워야 함.
        # 수정: 위에서 if forced_week_num: ... else: ... 로 분기했으므로 OK.
        # 단, notices 체크는 forced_week_num이 있어도 수행해야 할까? 
        # 사용자가 --week 3을 줬는데 notices 폴더면? -> 사용자의 의도(--week 3)를 따르는게 맞음.
        # 따라서, forced_week_num이 있으면 notices 체크도 무시하고 week_3으로 감.
        
        # 다만 file_type_cd 결정 로직은 week override와 별개로 동작해야 함.
        if forced_week_num is not None:
            # 강제 주차 지정 시, 파일 타입은 확장자 기반으로만 결정 (notices 폴더여도 강제로 주차 지정하면 일반 강의자료로 취급)
            # 혹은 notices 폴더면 N타입으로 하되 주차만 바꿀지? 
            # 보통 --week 옵션은 '강의자료'를 올릴 때 쓸 것이므로, notices 로직을 무시하는게 깔끔함.
            file_type_cd = "V" if ext.lower() in video_exts else "M"
        # (else 블록은 위에서 이미 처리됨)
        
        payload = {
            "cls_id": cls_id,
            "user_id": user_id,
            "file_type_cd": file_type_cd,
            "move_to_storage": True,
            "week_num": week_num,
            "files": {
                "orgFilename": filename,
                "extName": ext,
                "savedPathname": os.path.abspath(file_path),
                "size": file_size
            }
        }

        print(f"Processing: {filename} (Type: {file_type_cd}, Week: {week_num})")
        
        # API 호출
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            print(f"[성공] {filename} -> {response.json().get('message')}")
            if response.json().get('result'):
                print(f"       Result: {response.json().get('result')}")
        else:
            print(f"[실패] {filename} -> Status: {response.status_code}")
            print(f"       Body: {response.text}")

    except Exception as e:
        print(f"[에러] {file_path} 처리 중 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description="로컬 파일을 API를 통해 재임베딩합니다.")
    parser.add_argument("--dir", required=True, help="처리할 대상 디렉토리 경로 (절대 경로 권장)")
    parser.add_argument("--cls_id", required=True, help="강의실 ID (cls_id)")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API URL (기본값: {DEFAULT_API_URL})")

    parser.add_argument("--user", default=DEFAULT_USER_ID, help=f"요청자 ID (기본값: {DEFAULT_USER_ID})")
    parser.add_argument("--week", type=int, help="주차 강제 지정 (지정 시 자동 감지 무시)")
    parser.add_argument("--pattern", help="파일명 매칭을 위한 정규표현식 (예: 'week_1.*\.pdf')")
    
    args = parser.parse_args()
    
    target_dir = args.dir
    cls_id = args.cls_id
    
    # 대상이 파일인 경우
    if os.path.isfile(target_dir):
        process_file(target_dir, cls_id, args.url, args.user, args.week)
        count = 1
    # 대상이 디렉토리인 경우
    elif os.path.isdir(target_dir):
        count = 0 # Initialize count for directory processing
        # 디렉토리 순회 (재귀적)
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                # 숨김 파일 제외
                if file.startswith('.'):
                    continue
                
                # 정규식 필터링
                if args.pattern:
                    # NFD(MacOS) -> NFC(Standard) 정규화 후 매칭 시도
                    normalized_file = unicodedata.normalize('NFC', file)
                    normalized_pattern = unicodedata.normalize('NFC', args.pattern)
                    
                    if not re.search(normalized_pattern, normalized_file):
                        continue
                    
                file_path = os.path.join(root, file)
                process_file(file_path, cls_id, args.url, args.user, args.week)
                count += 1
    else:
        print(f"오류: 유효한 파일 또는 디렉토리가 아닙니다: {target_dir}")
        sys.exit(1)
            
    print("--------------------------------")
    print(f"작업 완료: 총 {count}개 파일 처리됨")

if __name__ == "__main__":
    main()
