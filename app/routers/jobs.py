"""파인튜닝 잡 엔드포인트 (API_SPEC.md §3·4·5).

- POST   /jobs              잡 제출 (201)
- GET    /jobs/{jobId}      잡 상태 조회
- GET    /jobs/{jobId}/logs SSE 로그 스트리밍
- GET    /jobs/{jobId}/model 파인튜닝된 모델 가중치 다운로드
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..catalog import get_model_by_id
from ..jobs_engine import manager
from ..schemas import JobRequest, JobResponse

router = APIRouter()


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(body: JobRequest):
    if get_model_by_id(body.backboneModelId) is None:
        return _error(422, f"지원하지 않는 모델 ID: {body.backboneModelId}", "MODEL_NOT_SUPPORTED")
    job = manager.create(body)
    return JSONResponse(
        JobResponse(jobId=job.job_id, status=job.status, createdAt=job.created_at).model_dump(
            exclude_none=True
        ),
        status_code=201,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        return _error(404, f"잡을 찾을 수 없습니다: {job_id}", "JOB_NOT_FOUND")
    return JSONResponse(job.to_response().model_dump(exclude_none=True))


@router.get("/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    job = manager.get(job_id)
    if job is None:
        return _error(404, f"잡을 찾을 수 없습니다: {job_id}", "JOB_NOT_FOUND")
    return StreamingResponse(
        manager.subscribe(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/model")
async def download_model(job_id: str):
    job = manager.get(job_id)
    if job is None:
        return _error(404, f"잡을 찾을 수 없습니다: {job_id}", "JOB_NOT_FOUND")
    if job.status != "completed" or job.model_path is None or not job.model_path.exists():
        return _error(404, "아직 모델 산출물이 준비되지 않았습니다.", "MODEL_NOT_READY")
    return FileResponse(
        path=str(job.model_path),
        filename=job.model_path.name,
        media_type="application/octet-stream",
    )


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, "details": {}}, status_code=status)
