import sys
import os
import asyncio
import asyncpg
import datetime

# 상위 폴더(backend)를 path에 추가하여 config, scripts 접근 가능하도록 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import DATABASE_CONFIG
from scripts.sync_lms_files import sync_lms

# API URL
DEFAULT_API_URL = "http://localhost:8085"

async def main():
    print(f"[{datetime.datetime.now()}] Daily Batch LMS Sync Started")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        # rag_yn이 'N' (신규) 또는 'Y' (활성) 인 모든 강의 조회
        # 단, user_div가 'O10' (교수)인 경우만 대상
        query = """
            SELECT c.cls_id, c.user_id, c.cls_yr, c.cls_smt, c.rag_yn 
            FROM cls_mst c
            INNER JOIN user_mst u ON c.user_id = u.user_id
            WHERE c.rag_yn IN ('N', 'Y')
            AND u.user_div = 'O10'
            ORDER BY c.ins_dt DESC
        """
        
        rows = await conn.fetch(query)
        print(f"Total {len(rows)} classes found for sync.")
        
        for row in rows:
            cls_id = row['cls_id']
            user_id = row['user_id']
            cls_yr = row['cls_yr']
            cls_smt = row['cls_smt']
            rag_yn = row['rag_yn']
            
            print(f"\n>>> Processing Class: {cls_id} ({user_id}) [RAG: {rag_yn}]")
            
            try:
                # 동기화 스크립트 호출 (성공여부, 처리건수 반환)
                success, count = sync_lms(user_id, cls_id, cls_yr, cls_smt, DEFAULT_API_URL)
                
                # 성공했고, 기존 상태가 신규('N')였던 경우 'Y'로 업데이트
                if success and rag_yn == 'N':
                    update_query = """
                        UPDATE cls_mst 
                        SET rag_yn = 'Y', upd_dt = NOW() 
                        WHERE cls_id = $1
                    """
                    await conn.execute(update_query, cls_id)
                    print(f" -> Updated rag_yn to 'Y' for {cls_id}")
                elif not success:
                    print(f" -> Failed to sync {cls_id}. Keeping rag_yn as '{rag_yn}'")
                    
                # [부하 분산 최적화]
                # 처리한 파일이 있는 경우에만 대기 (STT 작업 폭주 방지)
                if count > 0:
                    print(f" -> Processed {count} files. Waiting 10 seconds...")
                    await asyncio.sleep(10)
                else:
                    print(" -> No files processed. Skipping wait.")
                    
            except Exception as e:
                print(f"Error processing class {cls_id}: {e}")
                
        await conn.close()
        
    except Exception as e:
        print(f"Fatal Batch Error: {e}")

    print(f"[{datetime.datetime.now()}] Batch Finished")

if __name__ == "__main__":
    asyncio.run(main())
