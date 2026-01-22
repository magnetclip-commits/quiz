# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monorepo for an AI-powered educational tutoring platform with two main systems:
- **HLTA** (Teaching Assistant): Quiz generation, course material embedding, and instructor support
- **HLtutor**: Student-facing tutoring platform with real-time chat and discussion features

Primary languages: Python (FastAPI), Kotlin (Spring Boot), React

## Build and Run Commands

### HLTA Backend (FastAPI + Celery)
```bash
# Development with Docker Compose (recommended)
cd hlta
docker-compose up -d

# Manual run (requires Redis and ChromaDB)
cd hlta/backend
pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8085

# Celery worker
celery -A tasks worker --loglevel=info
```

### HLtutor Full Stack
```bash
cd hltutor
docker-compose up -d

# Individual services:
# Python API: port 8086
# Kotlin API: port 8111
# React frontend: port 3001
```

### Kotlin Backend (Spring Boot)
```bash
cd hltutor/hltutor-kotlin
gradle build
gradle bootRun --args='--spring.profiles.active=local'
gradle test
```

### React Frontend
```bash
cd hltutor/hltutor-react
npm install
npm start     # development
npm run build # production
npm test
```

## Architecture

### Service Ports
| Service | Port |
|---------|------|
| HLTA API (FastAPI) | 8085 |
| HLTA ChromaDB | 8002 |
| HLtutor Python API | 8086 |
| HLtutor Kotlin API | 8111 |
| HLtutor React | 3001 |
| HLtutor ChromaDB | 8003 |
| HLTA Redis | 6379 (internal) |
| HLtutor Redis | 6380 |

### Key Directories

**HLTA Backend (`hlta/backend/`):**
- `server.py` - FastAPI app entry point with async DB pool
- `routers/` - API endpoints (agent.py, file.py, users.py, student.py)
- `aita/` - AI quiz generation module with course-specific YAML profiles
- `embedding_*.py` - Vector embedding pipelines (m/n/v variants for different content)
- `tasks.py` - Celery task definitions and queue configuration
- `llm_factory.py` - LLM provider factory (see below)
- `config.py` - Database and environment configuration

**HLtutor (`hltutor/`):**
- `hltutor-python/` - Python FastAPI backend (similar structure to HLTA)
- `hltutor-kotlin/` - Spring Boot backend with WebSocket, JPA, batch processing
- `hltutor-react/` - React 19 frontend with Material-UI

### LLM Provider Configuration

The `llm_factory.py` supports multiple providers via `model_provider` setting:

| Code | Provider | Model |
|------|----------|-------|
| `gpt4o` | OpenAI | gpt-4o |
| `gpt4om` | OpenAI | gpt-4o-mini |
| `gpt41` | OpenAI | gpt-4.1 |
| `gpt5` | OpenAI | gpt-5 |
| `gpto3` | OpenAI | o3-2025-04-16 |
| `gpto3p` | OpenAI | o3-pro-2025-06-10 |
| `gmn25f` | Google | gemini-2.5-flash |
| `gmn25` | Google | gemini-2.5-pro |
| `cld4o` | Anthropic | claude-opus-4 |
| `pplx` | Perplexity | sonar-reasoning |
| `pplxp` | Perplexity | sonar-pro |

### Data Flow

1. **Document Processing**: Files uploaded → Celery queue → embedding_*.py → ChromaDB
2. **Quiz Generation**: Request → aita/quiz_create_v*.py → LLM (via llm_factory) → structured response
3. **Student Chat**: React → WebSocket/REST → Kotlin/Python API → LLM → ChromaDB (RAG)

### Database Configuration

Three database connections configured in `config.py`:
- `DATABASE_CONFIG` - Primary HLTA PostgreSQL
- `DATABASE2_CONFIG` - Legacy Oracle database
- `DATABASE3_CONFIG` - HLtutor PostgreSQL

### Environment Variables

Required API keys (set in `.env` files):
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `CLAUDE_API_KEY`
- `PERPLEXITY_API_KEY`

Infrastructure:
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`

## Development Notes

- Primary documentation language is Korean
- Both networks share `hlta_network` for inter-service communication
- ChromaDB uses persistent storage with health checks
- Async processing uses Celery with Redis broker
- Course profiles for quiz generation are YAML files in `aita/cls_*.yaml`
