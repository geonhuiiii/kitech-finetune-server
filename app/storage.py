"""업로드 파일을 디스크에 저장하는 청크 업로드 스토리지.

프론트엔드(DatasetUploadStep.tsx)는 Content-Range 헤더로 2MB 청크를 순차 전송합니다.
세션 디렉터리: {DATA_DIR}/uploads/{uploadId}/{role}/{filename}
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from .config import settings

_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._\-가-힣]")


@dataclass
class ChunkRange:
    start: int
    end: int
    total: int


def parse_content_range(header: str | None) -> ChunkRange | None:
    if not header:
        return None
    m = _CONTENT_RANGE_RE.match(header.strip())
    if not m:
        return None
    return ChunkRange(start=int(m.group(1)), end=int(m.group(2)), total=int(m.group(3)))


def sanitize_filename(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name)[:200]
    return cleaned or "file"


@dataclass
class UploadSession:
    upload_id: str
    received: dict[str, int] = field(default_factory=dict)  # "role/filename" -> bytes
    totals: dict[str, int] = field(default_factory=dict)


class UploadStore:
    def __init__(self) -> None:
        self._sessions: dict[str, UploadSession] = {}
        self._lock = Lock()

    def session_dir(self, upload_id: str) -> Path:
        return settings.uploads_dir / sanitize_filename(upload_id)

    def write_chunk(
        self,
        upload_id: str,
        role: str,
        filename: str,
        data: bytes,
        chunk: ChunkRange | None,
    ) -> tuple[str, int, int, bool]:
        """청크(또는 전체 파일)를 디스크에 기록하고 (filename, received, total, isComplete) 반환."""
        safe_role = sanitize_filename(role)
        safe_name = sanitize_filename(filename)
        role_dir = self.session_dir(upload_id) / safe_role
        role_dir.mkdir(parents=True, exist_ok=True)
        dest = role_dir / safe_name

        key = f"{safe_role}/{safe_name}"

        if chunk is None:
            # 단일 파일 업로드
            dest.write_bytes(data)
            received = len(data)
            total = len(data)
            is_complete = True
        else:
            # 청크 업로드 — start 오프셋에 기록
            mode = "wb" if chunk.start == 0 else "r+b"
            if mode == "r+b" and not dest.exists():
                dest.touch()
            with open(dest, mode) as fh:
                fh.seek(chunk.start)
                fh.write(data)
            received = chunk.end + 1
            total = chunk.total
            is_complete = received >= total

        with self._lock:
            sess = self._sessions.setdefault(upload_id, UploadSession(upload_id=upload_id))
            sess.received[key] = received
            sess.totals[key] = total

        return safe_name, received, total, is_complete

    def cleanup_stale(self) -> None:
        """TTL 초과 세션 디렉터리 정리."""
        root = settings.uploads_dir
        if not root.exists():
            return
        ttl = settings.UPLOAD_TTL_HOURS * 3600
        now = time.time()
        for entry in root.iterdir():
            try:
                if not entry.is_dir():
                    continue
                if now - entry.stat().st_mtime > ttl:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue


store = UploadStore()
