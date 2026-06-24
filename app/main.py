"""KITECH 파인튜닝 백엔드 서버 (FastAPI).

프론트엔드(RepositoryFinetuneChatbot)의 FINETUNE_API_SERVER_URL 로 연동되는 외부 서버.
구현 엔드포인트는 src/app/api/repositories/finetune-recommend/API_SPEC.md 를 따릅니다.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import jobs, recommend, upload
from .storage import store

app = FastAPI(
    title="KITECH Finetune Server",
    version="1.0.0",
    description="제조 AI 데이터셋 파인튜닝 백엔드 (모델 추천 / 업로드 / 학습 잡).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    settings.ensure_dirs()
    store.cleanup_stale()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "kitech-finetune-server"}


app.include_router(recommend.router)
app.include_router(upload.router)
app.include_router(jobs.router)
