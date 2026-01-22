"""
사용자가 직접 수정한 프로필(프롬프트)을 DB에 반영하고
해당 내용을 YAML 파일로 저장하는 스크립트.

- 교수자가 직접 수정한 경우에만 사용
- DB 저장 시 prof_mod_yn = 'Y', prof_mod_dt 갱신됨
- LLM 자동 저장과 구분하기 위해 sp_aita_profile_save를 6개 인자로 호출함

필수 입력값:
  user_id, profile_type, profile_text(프로필 변경 내용)

예시 실행:
  python prof_profile_update.py 99999 USR "3교수자 공통 프로필 내용입니다."
  # 인자를 생략하면 main()의 하드코딩된 기본값을 사용
"""

import asyncio
import os
import sys
import textwrap
from pathlib import Path

import asyncpg
import yaml

# 프로젝트 루트의 config 모듈을 찾기 위한 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import DATABASE_CONFIG  # PostgreSQL

BASE_DIR = Path(__file__).resolve().parent
YAML_ROOT = BASE_DIR / "yaml"

# 허용되는 프로필 타입 (DB 정책과 동일해야 함)
VALID_PROFILE_TYPES = {"USR", "CLS", "SYS"}

# .env 파일 로드 (선택적)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. 필요 시 pip install python-dotenv로 설치하세요.")


async def save_prompt_to_db(user_id: str, profile_type: str, profile_text: str) -> None:
    """
    교수자가 수정한 프로필을 DB에 저장한다.

    - sp_aita_profile_save(6개 인자)를 호출하여
      저장 출처를 'PROF'로 명시한다.
    - DB 트랜잭션 내에서 실행되어
      실패 시 YAML 파일이 생성되지 않도록 보장한다.
    """
    if profile_type not in VALID_PROFILE_TYPES:
        raise ValueError(
            f"Invalid profile_type: {profile_type} "
            f"(allowed: {sorted(VALID_PROFILE_TYPES)})"
        )

    file_path = f"/profiles/{profile_type}/{user_id}.yaml"

    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        async with conn.transaction():
            # 교수자 수정 저장 (PROF)
            await conn.execute(
                "CALL public.sp_aita_profile_save($1, $2, $3, $4, $5, $6)",
                user_id,
                "*",
                profile_type,
                profile_text,
                file_path,
                "PROF",
            )
    except Exception as e:
        # 운영 로그에서 원인 추적이 가능하도록 정보 포함
        print(
            f"[ERROR] 교수자 프로필 DB 반영 실패 "
            f"(user_id={user_id}, profile_type={profile_type})\n{e}"
        )
        raise
    finally:
        await conn.close()


def save_yaml_atomic(user_id: str, profile_type: str, profile_text: str) -> Path:
    """
    YAML 파일을 atomic 하게 저장한다.

    - 임시 파일(.tmp)에 먼저 작성
    - rename(replace)으로 실제 파일 교체
    - DB 저장 성공 이후에만 호출되어야 함
    - profile_text를 persona 필드로 변환하여 YAML 형식에 맞게 저장
    - 기존 YAML 파일이 있으면 name 필드는 유지하고 persona만 업데이트
    """
    target_dir = YAML_ROOT / profile_type
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{user_id}.yaml"
    temp_path = output_path.with_suffix(".yaml.tmp")

     # 기존 YAML 파일이 있으면 읽어서 name 필드 유지
    existing_data = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                # dict 형태가 아니면(문자열/리스트/None 등) name 유지 불가 → 빈 dict로 처리
                existing_data = loaded if isinstance(loaded, dict) else {}
        except Exception as e:
            print(f"[WARNING] 기존 YAML 파일 읽기 실패, 새로 생성합니다: {e}")
            existing_data = {}


    # YAML literal block scalar 형식으로 저장하기 위한 Dumper
    class _LiteralDumper(yaml.SafeDumper):
        """멀티라인 문자열을 YAML literal 블록(|)으로 출력"""

    def _repr_str(dumper, data):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _LiteralDumper.add_representer(str, _repr_str)

    def _normalize(text: str) -> str:
        """프롬프트 재사용을 위해 들여쓰기 제거 및 트리밍"""
        return textwrap.dedent(text or "").strip()

    # YAML 데이터 구성: 기존 name 필드는 유지, persona는 profile_text로 업데이트
    data = {
        "name": _normalize(existing_data.get("name", "")),
        "persona": _normalize(profile_text),
    }

    # YAML 형식으로 변환하여 저장
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=_LiteralDumper,
            allow_unicode=True,
            sort_keys=False,
            width=120,
            default_flow_style=False,
        )

    temp_path.replace(output_path)
    return output_path


async def run(user_id: str, profile_type: str, profile_text: str) -> Path:
    """
    전체 실행 흐름:
      1) DB에 교수자 수정 내용 저장
      2) 성공 시 YAML 파일 생성
    """
    await save_prompt_to_db(user_id, profile_type, profile_text)
    return save_yaml_atomic(user_id, profile_type, profile_text)


def main() -> None:
    # 인자를 주면 CLI 입력 사용, 없으면 하드코딩 기본값 사용
    if len(sys.argv) >= 4:
        user_id = sys.argv[1]
        profile_type = sys.argv[2]
        profile_text = sys.argv[3]
    else:
        user_id = "99999"
        profile_type = "USR"
        profile_text = "3교수자 공통 프로필 내용입니다."

    output_path = asyncio.run(run(user_id, profile_type, profile_text))
    print(f"DB 반영 및 YAML 생성 완료: {output_path}")


if __name__ == "__main__":
    main()
