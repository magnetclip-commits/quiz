'''
'''
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import jwt
import os
from routers import users, student, agent
from routers import file
# from routers import auth, chat, openbadge, users, board, learn, student, quiz, prompt
# from routers import ragtest
# from routers import rag_ready
from ocr_testpaper import imagePreprocessor, ocr_gpt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
import asyncpg
from contextlib import asynccontextmanager
from config import DATABASE_CONFIG # DB 접속 정보 임포트
import time

async def init_connection(conn):
    """
    새로운 DB 커넥션이 생성될 때마다 실행되는 함수
    이 커넥션의 타임존을 'Asia/Seoul'로 설정
    """
    print(f"DB 커넥션 [{conn.get_server_pid()}] 생성: 타임존을 Asia/Seoul로 설정합니다.")
    await conn.execute("SET TIME ZONE 'Asia/Seoul'")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 시작과 종료 시점에 DB 커넥션 풀을 관리
    - 시작: DB 풀을 생성하고 app.state에 저장
    - 종료: 저장된 DB 풀을 안전하게 close
    """
    # --- 앱 시작 시 실행 ---
    print("[Lifespan] 앱 시작: DB 커넥션 풀을 생성합니다.")
    app.state.db_pool = await asyncpg.create_pool(
        **DATABASE_CONFIG, min_size=5, max_size=50, init=init_connection 
    )
    
    yield # 앱이 실행되는 동안 이 지점에서 대기합니다.

    # --- 앱 종료 시 실행 ---
    print("[Lifespan] 앱 종료: DB 커넥션 풀을 닫습니다.")
    if app.state.db_pool:
        await app.state.db_pool.close()

# from routers import evaluate
SECRET_KEY = "my-super-secret-key-1234567890123456"
ALGORITHM = "HS256"

# # --- 로깅 설정 추가 ---
# LOGGING_CONFIG = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": {
#             "()": "uvicorn.logging.DefaultFormatter",
#             "fmt": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#             "use_colors": None,
#         },
#         "access": {
#             "()": "uvicorn.logging.AccessFormatter",
#             "fmt": '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
#         },
#     },
#     "handlers": {
#         "default": {
#             "formatter": "default",
#             "class": "logging.StreamHandler",
#             "stream": "ext://sys.stderr",
#         },
#         "access": {
#             "formatter": "access",
#             "class": "logging.StreamHandler",
#             "stream": "ext://sys.stdout",
#         },
#     },
#     "loggers": {
#         "": {"handlers": ["default"], "level": "INFO"},
#         "uvicorn.error": {"level": "INFO"},
#         "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
#     },
# }
    
app = FastAPI(lifespan=lifespan)
# app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is truly alive"}

@app.post("/")
async def post_root():
    return {"status": "ok", "message": "Service is alive"}

@app.get("/")
async def read_root():
   return {"message": "Hello"}


app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(student.router, prefix="/student", tags=["Student"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])
app.include_router(file.router, prefix="/file", tags=["File"])
# app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
# app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
# app.include_router(file.router, prefix="/api/file", tags=["File"])
# app.include_router(users.router, prefix="/api/users", tags=["Users"])
# app.include_router(learn.router, prefix="/api/learn", tags=["Learn"])
# app.include_router(student.router, prefix="/api/student", tags=["Student"])
# app.include_router(openbadge.router, prefix="/api/openbadge", tags=["OpenBadge"])
# app.include_router(board.router, prefix="/api/board", tags=["Board"])
# app.include_router(quiz.router, prefix="/api/quiz", tags=["Quiz"])
# app.include_router(prompt.router, prefix="/api/prompt", tags=["Prompt"])
# # app.include_router(evaluate.router, prefix="/api/evaluate", tags=["Evaluate"])
# # app.include_router(ragtest.router, prefix="/api/ragtest", tags=["Ragtest"])
# app.include_router(rag_ready.router, prefix="/api/rag", tags=["RAG"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

