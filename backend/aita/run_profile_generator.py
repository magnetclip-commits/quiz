"""
DB에서 profile_body가 NULL인 레코드를 조회하여
prof_profile_generator.py와 cls_profile_generator.py를 실행하는 스크립트.

사용법:
  python run_profile_generator.py
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg

# 패키지 실행(-m)과 스크립트 실행 모두 동작하도록 경로 보정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DATABASE_CONFIG  # PostgreSQL

# 패키지/모듈 import 호환
try:
    from .prof_profile_generator import run as prof_run
    from .cls_profile_generator import run as cls_run
except ImportError:
    from prof_profile_generator import run as prof_run
    from cls_profile_generator import run as cls_run


async def fetch_user_ids_without_profile():
    """profile_body가 NULL인 user_id 목록 조회"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        query = """
        SELECT user_id 
        FROM aita_profile_mst
        WHERE profile_type = 'USR'
          AND profile_body IS NULL
        """
        rows = await conn.fetch(query)
        return [row['user_id'] for row in rows]
    finally:
        await conn.close()


async def fetch_cls_ids_without_profile():
    """profile_body가 NULL인 cls_id 목록 조회"""
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        query = """
        SELECT cls_id 
        FROM aita_profile_mst
        WHERE profile_type = 'CLS'
          AND profile_body IS NULL
        """
        rows = await conn.fetch(query)
        return [row['cls_id'] for row in rows]
    finally:
        await conn.close()


async def run_prof_profile_generators(user_ids):
    """user_id 목록에 대해 prof_profile_generator.py 실행"""
    if not user_ids:
        print("생성할 교수자 프로필이 없습니다.")
        return
    
    print(f"\n{'='*60}")
    print(f"교수자 프로필 생성 시작: {len(user_ids)}개")
    print(f"{'='*60}")
    
    success_count = 0
    fail_count = 0
    
    for idx, user_id in enumerate(user_ids, 1):
        print(f"\n[{idx}/{len(user_ids)}] 교수자 프로필 생성 중: user_id={user_id}")
        try:
            # prof_profile_generator의 run 함수는 async 함수이므로 await로 호출
            await prof_run(user_id)
            print(f"성공: user_id={user_id}")
            success_count += 1
        except Exception as e:
            print(f"실패: user_id={user_id}, 오류: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"교수자 프로필 생성 완료: 성공 {success_count}개, 실패 {fail_count}개")
    print(f"{'='*60}")


async def run_cls_profile_generators(cls_ids):
    """cls_id 목록에 대해 cls_profile_generator.py 실행"""
    if not cls_ids:
        print("생성할 강의 프로필이 없습니다.")
        return
    
    print(f"\n{'='*60}")
    print(f"강의 프로필 생성 시작: {len(cls_ids)}개")
    print(f"{'='*60}")
    
    success_count = 0
    fail_count = 0
    
    for idx, cls_id in enumerate(cls_ids, 1):
        print(f"\n[{idx}/{len(cls_ids)}] 강의 프로필 생성 중: cls_id={cls_id}")
        try:
            # cls_profile_generator의 run 함수는 async 함수이므로 await로 호출
            await cls_run(cls_id)
            print(f"성공: cls_id={cls_id}")
            success_count += 1
        except Exception as e:
            print(f"실패: cls_id={cls_id}, 오류: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"강의 프로필 생성 완료: 성공 {success_count}개, 실패 {fail_count}개")
    print(f"{'='*60}")


async def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("프로필 생성 스크립트 시작")
    print("="*60)
    
    # DB에서 데이터 조회
    print("\n[1단계] DB에서 프로필이 없는 레코드 조회 중...")
    user_ids = await fetch_user_ids_without_profile()
    cls_ids = await fetch_cls_ids_without_profile()
    
    print(f"  - 교수자 프로필 생성 필요: {len(user_ids)}개")
    print(f"  - 강의 프로필 생성 필요: {len(cls_ids)}개")
    
    if not user_ids and not cls_ids:
        print("\n생성할 프로필이 없습니다.")
        return
    
    # 교수자 프로필 생성
    if user_ids:
        await run_prof_profile_generators(user_ids)
    
    # 강의 프로필 생성
    if cls_ids:
        await run_cls_profile_generators(cls_ids)
    
    print("\n" + "="*60)
    print("모든 프로필 생성 작업 완료")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())