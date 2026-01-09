# root working dirextory; /opt/hlta

## . directory
backend; 조교 문항 생성을 위한 백엔드 코드
docker-compose.yml; fastapi, chromaDB, 비동기 처리를 위한 컨테이너 정의 파일 
.env; 각종 설정 파라미터 및 인증 정보
logs; log용 디렉토리
migration; 튜터의 데이터 마이그레이션 처리
readme.md; 본 파일

### sub directory; /backend
aita; 문제 생성 코드 및 문제 생성을 위한 작성한 교과목별 프로파일 
api.Dockerfile; api 컨테이너 구축을 위한 환경 설정 파일
check_chroma.py
chromadb_source
config.py
config.py~
consumer_m.py; 자료 임베딩시 비동기 처리를 위한 제어 코드
consumer_n.py; 자료 임베딩시 비동기 처리를 위한 제어 코드
consumer_v.py; 자료 임베딩시 비동기 처리를 위한 제어 코드
embedding_m.py; 자료 임베딩 처리의 실체 코드
embedding_n.py; 자료 임베딩 처리의 실체 코드
embedding_v.py; 자료 임베딩 처리의 실체 코드
etl; 튜터용 RDB의 ETL 처리
get_lms_file_dwld_notice.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
get_lms_file_dwld.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
get_lms_file_dwld_week_m.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
get_lms_file_dwld_week_v.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
get_lms_file_list.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
get_lms_notice_list.py; 임베딩한 스마트리드 자료의 주별 리스트 취득 코드
hallym_smart_lead; (튜터) 스마트리드 접속을 위한 각종 처리
llm_factory.py; llm 모델을 지정
logging.ini; log 저장을 위한 설정 파일
logs; log저장용 디렉토리 
migrate.py; 튜터의 chromadb를 마이그레이션 하기 위한 처리
ocr_testpaper; 답안지 OCR 처리
output_images; OCR 처리 후 이미지 저장
requirements.txt; 필요한 패키지 인스톨을 위한 패키지 정의
routers; fastapi 경로 정의
server.py; fastapi 정의
tasks.py; 비동기 처리를 위한 환경 설정 및 파라미터 정의 

### sub directory; /migration
check_chroma.py; chromaDB 디렉토리 체크용
merge_tutor_db.py; 튜터 chromaDB를 조교 chromaDB에 merge
migrate.py; 튜터용 chromaDB 초기  이관용
verify_search.py; 테스트용 코드
