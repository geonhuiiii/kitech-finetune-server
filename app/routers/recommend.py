"""POST /recommend — 백본 모델 추천 (API_SPEC.md §1)."""
from __future__ import annotations

from fastapi import APIRouter

from ..catalog import get_recommended_models
from ..schemas import RecommendRequest, RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest) -> RecommendResponse:
    models = get_recommended_models(body.dataType, body.taskType)
    return RecommendResponse(models=models, isFallback=False)
