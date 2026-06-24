"""POST /upload — 데이터셋 청크 업로드 (API_SPEC.md §2).

프론트엔드는 multipart/form-data 로 file, uploadId, role 을 보내고
청크 업로드 시 Content-Range 헤더를 함께 전송합니다.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from ..config import settings
from ..schemas import UploadResponse
from ..storage import parse_content_range, store

router = APIRouter()


@router.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    uploadId: str = Form(...),
    role: str = Form("file"),
    content_range: str | None = Header(default=None, alias="content-range"),
) -> JSONResponse:
    chunk = parse_content_range(content_range)

    if chunk is not None and chunk.total > settings.MAX_FILE_SIZE:
        return _error(
            413,
            f"파일이 너무 큽니다. 최대 {settings.MAX_FILE_SIZE // (1024 ** 3)}GB",
            "UPLOAD_TOO_LARGE",
        )

    data = await file.read()

    if chunk is None and len(data) > settings.MAX_FILE_SIZE:
        return _error(
            413,
            f"파일이 너무 큽니다. 최대 {settings.MAX_FILE_SIZE // (1024 ** 3)}GB",
            "UPLOAD_TOO_LARGE",
        )

    filename, received, total, is_complete = store.write_chunk(
        upload_id=uploadId,
        role=role,
        filename=file.filename or "file",
        data=data,
        chunk=chunk,
    )

    payload = UploadResponse(
        uploadId=uploadId,
        filename=filename,
        role=role,
        received=received,
        total=total,
        isComplete=is_complete,
    )
    return JSONResponse(payload.model_dump())


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, "details": {}}, status_code=status)
