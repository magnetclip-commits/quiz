import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", 5433)),
    "user": os.getenv("DATABASE_USER", "idta"),
    "password": os.getenv("DATABASE_PASSWORD", "password"),
    "database": os.getenv("DATABASE_NAME", "hlta"),
}

DATABASE2_CONFIG = {
    "user": os.getenv("DATABASE2_USER"),
    "password": os.getenv("DATABASE2_PASSWORD"),
    "dsn": os.getenv("DATABASE2_DSN"),
}

DATABASE3_CONFIG = {
    "host": os.getenv("DATABASE3_HOST", "localhost"),
    "port": int(os.getenv("DATABASE3_PORT", 5432)),
    "user": os.getenv("DATABASE3_USER", "user"),
    "password": os.getenv("DATABASE3_PASSWORD", "password"),
    "database": os.getenv("DATABASE3_NAME", "hltutor"),
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")


FRONTEND_URL = os.getenv("FRONTEND_URL")

REDIS_CONFIG = {
    "host":os.getenv("REDIS_HOST", "172.18.0.1"),
    "port":os.getenv("REDIS_PORT", 8083),
    "password":os.getenv("REDIS_PASSWORD", "admin1234!"),
    "db":os.getenv("REDIS_DB", 0)
}

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my-super-secret-key-1234567890123456")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
