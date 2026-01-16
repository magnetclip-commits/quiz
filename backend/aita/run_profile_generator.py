"""
DB에서 profile_body가 NULL인 레코드를 조회하여
prof_profile_generator.py와 cls_profile_generator.py를 실행하는 스크립트.

사용법:
  python run_profile_generator.py
"""

import asyncio
import sys
import time
from datetime import datetime
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


def log(msg: str):
    """cron/docker 환경에서도 즉시 찍히도록 flush=True"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def fmt_elapsed(seconds: float) -> str:
    """초를 사람이 읽기 쉬운 형태로 변환"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}시간 {m}분 {s:.2f}초"
    if m > 0:
        return f"{m}분 {s:.2f}초"
    return f"{s:.2f}초"


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
        return [row["user_id"] for row in rows]
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
        return [row["cls_id"] for row in rows]
    finally:
        await conn.close()


async def run_prof_profile_generators(user_ids):
    """user_id 목록에 대해 prof_profile_generator.py 실행"""
    if not user_ids:
        log("생성할 교수자 프로필이 없습니다.")
        return

    log("=" * 60)
    log(f"교수자 프로필 생성 시작: {len(user_ids)}개")
    log("=" * 60)

    success_count = 0
    fail_count = 0
    start_perf = time.perf_counter()

    for idx, user_id in enumerate(user_ids, 1):
        item_start = time.perf_counter()
        log(f"[{idx}/{len(user_ids)}] 교수자 프로필 생성 중: user_id={user_id}")

        try:
            # prof_profile_generator의 run 함수는 async 함수이므로 await로 호출
            await prof_run(user_id)
            elapsed_item = time.perf_counter() - item_start
            log(f"성공: user_id={user_id} (소요 {fmt_elapsed(elapsed_item)})")
            success_count += 1
        except Exception as e:
            elapsed_item = time.perf_counter() - item_start
            log(f"실패: user_id={user_id} (소요 {fmt_elapsed(elapsed_item)}), 오류: {str(e)}")
            fail_count += 1

    elapsed_total = time.perf_counter() - start_perf
    log("=" * 60)
    log(f"교수자 프로필 생성 완료: 성공 {success_count}개, 실패 {fail_count}개, 총 소요 {fmt_elapsed(elapsed_total)}")
    log("=" * 60)


async def run_cls_profile_generators(cls_ids):
    """cls_id 목록에 대해 cls_profile_generator.py 실행"""
    if not cls_ids:
        log("생성할 강의 프로필이 없습니다.")
        return

    log("=" * 60)
    log(f"강의 프로필 생성 시작: {len(cls_ids)}개")
    log("=" * 60)

    success_count = 0
    fail_count = 0
    start_perf = time.perf_counter()

    for idx, cls_id in enumerate(cls_ids, 1):
        item_start = time.perf_counter()
        log(f"[{idx}/{len(cls_ids)}] 강의 프로필 생성 중: cls_id={cls_id}")

        try:
            # cls_profile_generator의 run 함수는 async 함수이므로 await로 호출
            await cls_run(cls_id)
            elapsed_item = time.perf_counter() - item_start
            log(f"성공: cls_id={cls_id} (소요 {fmt_elapsed(elapsed_item)})")
            success_count += 1
        except Exception as e:
            elapsed_item = time.perf_counter() - item_start
            log(f"실패: cls_id={cls_id} (소요 {fmt_elapsed(elapsed_item)}), 오류: {str(e)}")
            fail_count += 1

    elapsed_total = time.perf_counter() - start_perf
    log("=" * 60)
    log(f"강의 프로필 생성 완료: 성공 {success_count}개, 실패 {fail_count}개, 총 소요 {fmt_elapsed(elapsed_total)}")
    log("=" * 60)


async def main():
    """메인 실행 함수"""
    start_wall_time = datetime.now()
    start_perf = time.perf_counter()

    log("=" * 60)
    log("프로필 생성 스크립트 시작")
    log(f"시작 시각 : {start_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # DB에서 데이터 조회
    log("[1단계] DB에서 프로필이 없는 레코드 조회 중...")
    user_ids = await fetch_user_ids_without_profile()
    cls_ids = await fetch_cls_ids_without_profile()

    log(f"교수자 프로필 생성 필요: {len(user_ids)}개")
    log(f"강의 프로필 생성 필요: {len(cls_ids)}개")

    if not user_ids and not cls_ids:
        end_wall_time = datetime.now()
        elapsed = time.perf_counter() - start_perf
        log("생성할 프로필이 없습니다.")
        log(f"종료 시각 : {end_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"총 소요 시간 : {fmt_elapsed(elapsed)}")
        log("=" * 60)
        return

    # 교수자 프로필 생성
    if user_ids:
        await run_prof_profile_generators(user_ids)

    # 강의 프로필 생성
    if cls_ids:
        await run_cls_profile_generators(cls_ids)

    end_wall_time = datetime.now()
    elapsed = time.perf_counter() - start_perf

    log("=" * 60)
    log("모든 프로필 생성 작업 완료")
    log(f"시작 시각 : {start_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"종료 시각 : {end_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"총 소요 시간 : {fmt_elapsed(elapsed)}")
    log("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
