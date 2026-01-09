import asyncio
import sys
import socket
import os
from pathlib import Path
import asyncpg
from dotenv import load_dotenv

# .env 파일 로드 (여러 경로 시도)
env_paths = [
    '/opt/hlta/.env',  # 컨테이너 환경에서 사용하는 경로
    Path(__file__).parent.parent / '.env',  # 상위 디렉토리의 .env
    Path(__file__).parent / '.env',  # 현재 디렉토리의 .env
    '.env',  # 현재 작업 디렉토리의 .env
]

env_loaded = False
for env_path in env_paths:
    if isinstance(env_path, Path):
        env_path_str = str(env_path)
    else:
        env_path_str = env_path
    
    if os.path.exists(env_path_str):
        load_dotenv(env_path_str)
        print(f"✅ .env 파일 로드 성공: {env_path_str}")
        env_loaded = True
        break

if not env_loaded:
    # 기본적으로 load_dotenv() 시도 (현재 디렉토리에서 찾기)
    load_dotenv()
    print("⚠️  명시적인 .env 파일을 찾지 못했습니다. 기본 경로에서 시도합니다.")

# 상위 디렉토리를 sys.path에 추가하여 config 모듈을 찾을 수 있도록 함
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_CONFIG, DATABASE3_CONFIG


def check_host_resolution(host):
    """호스트명 해석 가능 여부 확인"""
    try:
        ip_address = socket.gethostbyname(host)
        return True, ip_address
    except socket.gaierror as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_port_connectivity(host, port, timeout=3):
    """포트 연결 가능 여부 확인"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


async def test_database_connection(config, db_name):
    """데이터베이스 접속을 테스트하는 함수"""
    host = config.get('host', 'N/A')
    port = config.get('port', 'N/A')
    
    print(f"\n{'='*50}")
    print(f"[{db_name}] 접속 시도 중...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Database: {config.get('database', 'N/A')}")
    print(f"User: {config.get('user', 'N/A')}")
    print(f"{'='*50}")
    
    # 1단계: 호스트명 해석 확인
    if host != 'N/A':
        print(f"\n[진단 1단계] 호스트명 해석 확인 중...")
        resolved, result = check_host_resolution(host)
        if resolved:
            print(f"   ✅ 호스트명 해석 성공: {host} -> {result}")
        else:
            print(f"   ❌ 호스트명 해석 실패: {result}")
            print(f"   💡 해결 방법:")
            print(f"      - 호스트명이 올바른지 확인하세요")
            print(f"      - 네트워크 연결을 확인하세요")
            print(f"      - Docker 환경이라면 'host.docker.internal' 대신 'localhost' 또는 실제 IP를 사용하세요")
            print(f"      - /etc/hosts 파일에 호스트명이 등록되어 있는지 확인하세요")
            return False
        
        # 2단계: 포트 연결 확인
        if port != 'N/A':
            print(f"\n[진단 2단계] 포트 연결 확인 중...")
            port_available = check_port_connectivity(host, port)
            if port_available:
                print(f"   ✅ 포트 {port} 연결 가능")
            else:
                print(f"   ❌ 포트 {port} 연결 불가")
                print(f"   💡 해결 방법:")
                print(f"      - 데이터베이스 서버가 실행 중인지 확인하세요")
                print(f"      - 방화벽 설정을 확인하세요")
                print(f"      - 포트 번호가 올바른지 확인하세요")
                return False
    
    # 3단계: 데이터베이스 연결 시도
    try:
        print(f"\n[진단 3단계] 데이터베이스 연결 시도 중...")
        conn = await asyncpg.connect(**config)
        
        # 간단한 쿼리 실행하여 연결 확인
        version = await conn.fetchval('SELECT version()')
        current_db = await conn.fetchval('SELECT current_database()')
        current_user = await conn.fetchval('SELECT current_user')
        
        print(f"\n✅ [{db_name}] 접속 성공!")
        print(f"   - PostgreSQL 버전: {version.split(',')[0]}")
        print(f"   - 현재 데이터베이스: {current_db}")
        print(f"   - 현재 사용자: {current_user}")
        
        await conn.close()
        return True
        
    except asyncpg.exceptions.InvalidPasswordError as e:
        print(f"\n❌ [{db_name}] 접속 실패!")
        print(f"   오류 유형: 인증 실패 (잘못된 비밀번호)")
        print(f"   오류 메시지: {str(e)}")
        print(f"   💡 해결 방법: 비밀번호를 확인하세요")
        return False
    except asyncpg.exceptions.InvalidCatalogNameError as e:
        print(f"\n❌ [{db_name}] 접속 실패!")
        print(f"   오류 유형: 데이터베이스가 존재하지 않음")
        print(f"   오류 메시지: {str(e)}")
        print(f"   💡 해결 방법: 데이터베이스 이름을 확인하세요")
        return False
    except OSError as e:
        print(f"\n❌ [{db_name}] 접속 실패!")
        print(f"   오류 유형: 네트워크 오류")
        print(f"   오류 메시지: {str(e)}")
        if "Name or service not known" in str(e):
            print(f"   💡 해결 방법:")
            print(f"      - 호스트명을 확인하세요 (현재: {host})")
            print(f"      - Docker 환경이 아니라면 'host.docker.internal'을 'localhost'로 변경해보세요")
            print(f"      - 네트워크 연결을 확인하세요")
        elif "Connection refused" in str(e):
            print(f"   💡 해결 방법:")
            print(f"      - 데이터베이스 서버가 실행 중인지 확인하세요")
            print(f"      - 포트 번호를 확인하세요 (현재: {port})")
        elif "timed out" in str(e).lower():
            print(f"   💡 해결 방법:")
            print(f"      - 방화벽 설정을 확인하세요")
            print(f"      - 네트워크 연결을 확인하세요")
        return False
    except Exception as e:
        print(f"\n❌ [{db_name}] 접속 실패!")
        print(f"   오류 유형: {type(e).__name__}")
        print(f"   오류 메시지: {str(e)}")
        return False


async def main():
    """두 데이터베이스 접속을 모두 테스트"""
    print("\n" + "="*50)
    print("데이터베이스 접속 테스트 시작")
    print("="*50)
    
    # 환경 변수 확인 (디버깅용)
    print("\n[환경 변수 확인]")
    print(f"DB_HOST: {os.getenv('DB_HOST', '설정되지 않음')}")
    print(f"DB_PORT: {os.getenv('DB_PORT', '설정되지 않음')}")
    print(f"DB_NAME: {os.getenv('DB_NAME', '설정되지 않음')}")
    print(f"DB_USER: {os.getenv('DB_USER', '설정되지 않음')}")
    print(f"DATABASE3_HOST: {os.getenv('DATABASE3_HOST', '설정되지 않음')}")
    print(f"DATABASE3_PORT: {os.getenv('DATABASE3_PORT', '설정되지 않음')}")
    print(f"DATABASE3_NAME: {os.getenv('DATABASE3_NAME', '설정되지 않음')}")
    print(f"DATABASE3_USER: {os.getenv('DATABASE3_USER', '설정되지 않음')}")
    
    results = {}
    
    # DATABASE_CONFIG 테스트 (원본)
    print("\n" + "="*50)
    print("원본 DATABASE_CONFIG로 테스트")
    print("="*50)
    results['DATABASE_CONFIG (원본)'] = await test_database_connection(
        DATABASE_CONFIG, 
        "DATABASE_CONFIG (원본)"
    )
    
    # DATABASE_CONFIG 테스트 (HOST만 localhost로 변경)
    print("\n" + "="*50)
    print("HOST를 localhost로 변경한 DATABASE_CONFIG로 테스트")
    print("="*50)
    DATABASE_CONFIG_LOCALHOST = DATABASE_CONFIG.copy()
    DATABASE_CONFIG_LOCALHOST['host'] = 'localhost'
    print(f"원본 HOST: {DATABASE_CONFIG.get('host')} -> 변경된 HOST: localhost")
    
    results['DATABASE_CONFIG (localhost)'] = await test_database_connection(
        DATABASE_CONFIG_LOCALHOST, 
        "DATABASE_CONFIG (localhost)"
    )
    
    # DATABASE3_CONFIG 테스트
    print("\n" + "="*50)
    print("DATABASE3_CONFIG로 테스트")
    print("="*50)
    results['DATABASE3_CONFIG'] = await test_database_connection(
        DATABASE3_CONFIG, 
        "DATABASE3_CONFIG"
    )
    
    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    for db_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{db_name}: {status}")
    
    print("="*50)
    
    # 모든 접속이 성공했는지 확인
    if all(results.values()):
        print("\n🎉 모든 데이터베이스 접속이 성공했습니다!")
        return 0
    else:
        print("\n⚠️  일부 데이터베이스 접속에 실패했습니다.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
