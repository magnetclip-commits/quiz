import asyncio
import os
import sys
from pathlib import Path
import asyncpg

# Add parent directory to sys.path to find config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_CONFIG

# Test Professor Data Mapping
PROF_INFO = {
    "성시일": {"dpt_nm": "중국학과", "org_nm": "인문대학"},
    "박종민": {"dpt_nm": "미래융합스쿨", "org_nm": "대학"},
    "김은주": {"dpt_nm": "소프트웨어학부", "org_nm": "정보과학대학"}
}

TEST_DATA = [
    ("g260114001", "Hallym1!", "성시일", "무역결제론"),
    ("g260114002", "Hallym2!", "박종민", "재료과학개론II"),
    ("g260114003", "Hallym3!", "김은주", "창의코딩-모두의웹"),
    ("g260114004", "Hallym4!", "성시일", "무역결제론"),
    ("g260114005", "Hallym5!", "박종민", "재료과학개론II"),
    ("g260114006", "Hallym6!", "김은주", "창의코딩-모두의웹"),
    ("g260114007", "Hallym7!", "성시일", "무역결제론"),
    ("g260114008", "Hallym8!", "박종민", "재료과학개론II"),
    ("g260114009", "Hallym9!", "김은주", "창의코딩-모두의웹"),
    ("g260114010", "Hallym10!", "성시일", "무역결제론"),
]

async def register_test_accounts():
    conn = await asyncpg.connect(**DATABASE_CONFIG)
    try:
        async with conn.transaction():
            for i, (u_id, pw, name, cls_nm) in enumerate(TEST_DATA, 1):
                info = PROF_INFO[name]
                cls_id = f"G-2025-2-{i:03d}"
                
                # 1. Register User
                print(f"Registering User: {u_id} ({name})")
                await conn.execute("""
                    INSERT INTO user_mst (user_id, user_div, user_pw, user_nm, dpt_nm, org_nm, user_ctg)
                    VALUES ($1, 'PROF', $2, $3, $4, $5, 'INTE')
                    ON CONFLICT (user_id) DO UPDATE 
                    SET user_pw = EXCLUDED.user_pw, user_nm = EXCLUDED.user_nm
                """, u_id, pw, f"{name}(Test)", info["dpt_nm"], info["org_nm"])

                # 2. Register Class
                print(f"Registering Class: {cls_id} ({cls_nm})")
                await conn.execute("""
                    INSERT INTO cls_mst (cls_id, course_id, cls_nm, cls_sec, user_id, cls_yr, cls_smt, cls_grd, rag_yn, upload_type)
                    VALUES ($1, $2, $3, '01', $4, '2025', '2', '1', 'Y', 'M')
                    ON CONFLICT (cls_id) DO UPDATE
                    SET cls_nm = EXCLUDED.cls_nm, user_id = EXCLUDED.user_id
                """, cls_id, f"G-COURSE-{i:03d}", cls_nm, u_id)
                
            print("\nSuccessfully registered 10 test users and classes.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(register_test_accounts())
