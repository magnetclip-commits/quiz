# /opt/hlta/backend/api.Dockerfile

# 파이썬 공식 이미지를 기반으로 사용합니다.
FROM python:3.11-slim

# 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 시스템 의존성 설치 (PostgreSQL 연결에 필요한 라이브러리)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev 

RUN apt-get update && \
    apt-get install -y dumb-init

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    libglib2.0-0 \ 
    ffmpeg \ 
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    postgresql-client \
    iputils-ping \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 전체 복사 (auth.py의 최신 수정 내용 포함)
COPY . .

# FastAPI 애플리케이션이 수신 대기할 포트를 노출합니다.
EXPOSE 8085

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
# 컨테이너 시작 명령 (docker-compose.yml의 command가 이를 덮어쓸 것임)
CMD ["python","server.py"]
