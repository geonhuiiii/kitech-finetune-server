"""파인튜닝 잡 관리 + 학습 시뮬레이션 + SSE 로그 스트리밍.

⚠️ 현재 학습 단계는 시뮬레이션입니다. 실제 학습 파이프라인을 붙이려면
`_train` 코루틴 내부를 교체하세요 (예: torch/anomalib/ultralytics subprocess 실행).
잡 상태/로그/산출물 인터페이스는 그대로 두면 프론트엔드 변경 없이 동작합니다.
"""
from __future__ import annotations

import asyncio
import json
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

from .catalog import get_model_by_id
from .config import settings
from .schemas import JobRequest, JobResponse, JobStatus
from .storage import store
from .trainers.registry import get_trainer


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    job_id: str
    request: JobRequest
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now_iso)
    progress: int = 0
    estimated_seconds: int = 0
    logs: list[str] = field(default_factory=list)
    model_path: Optional[Path] = None
    error: Optional[str] = None
    # SSE 구독자 큐 + 종료 플래그
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _done: bool = False

    def to_response(self) -> JobResponse:
        download_url = None
        if self.status == "completed" and self.model_path is not None:
            base = settings.PUBLIC_BASE_URL.rstrip("/")
            path = f"/jobs/{self.job_id}/model"
            download_url = f"{base}{path}" if base else path
        return JobResponse(
            jobId=self.job_id,
            status=self.status,
            createdAt=self.created_at,
            progress=self.progress,
            estimatedSeconds=self.estimated_seconds,
            logs=self.logs[-50:],
            modelDownloadUrl=download_url,
            error=self.error,
        )


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def create(self, req: JobRequest) -> Job:
        job_id = f"job_{secrets.token_hex(6)}"
        job = Job(job_id=job_id, request=req)
        job.estimated_seconds = int(settings.SIM_TOTAL_EPOCHS * settings.SIM_EPOCH_SECONDS)
        self._jobs[job_id] = job
        asyncio.create_task(self._train(job))
        return job

    # ── 이벤트 발행/구독 (SSE) ──────────────────────────────────────────────
    def _emit(self, job: Job, event: dict) -> None:
        for q in list(job._subscribers):
            q.put_nowait(event)

    async def subscribe(self, job: Job) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        job._subscribers.append(queue)
        try:
            # 신규 구독자에게 누적 로그를 먼저 전달
            for line in job.logs:
                yield _sse({"type": "log", "message": line})
            if job._done:
                yield _terminal_event(job)
                return
            while True:
                event = await queue.get()
                yield _sse(event)
                if event.get("type") in ("complete", "error"):
                    return
        finally:
            if queue in job._subscribers:
                job._subscribers.remove(queue)

    # ── 학습 진입점: 실제 트레이너 우선, 없으면 시뮬레이션 폴백 ────────────────
    async def _train(self, job: Job) -> None:
        try:
            model = get_model_by_id(job.request.backboneModelId)
            model_name = model.name if model else job.request.backboneModelId

            job.status = "running"
            self._log(job, f"[init] 모델={model_name}, 태스크={job.request.dataType}/{job.request.taskType}/{job.request.subTaskType}")
            self._log(job, f"[init] uploadId={job.request.uploadId} 데이터 준비")

            trainer = get_trainer(job.request.dataType, job.request.taskType)
            if trainer is not None and trainer.is_available():
                self._log(job, "[init] 실제 학습 모드 (torch)")
                train_loss, val_loss = await self._run_real_training(job, trainer)
            else:
                if trainer is not None:
                    self._log(job, "[init] torch 의존성 미설치 → 시뮬레이션 모드로 폴백")
                else:
                    self._log(job, "[init] 해당 태스크 실제 트레이너 미구현 → 시뮬레이션 모드")
                train_loss, val_loss = await self._run_simulation(job)

            job.progress = 100
            job.estimated_seconds = 0
            job.status = "completed"
            self._log(job, f"[result] final train_loss={train_loss}, val_loss={val_loss}")
            self._finish(job, {
                "type": "complete",
                "modelDownloadUrl": job.to_response().modelDownloadUrl,
            })
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            self._log(job, f"[error] {exc}")
            self._finish(job, {"type": "error", "message": str(exc)})

    # ── 실제 학습: 블로킹 트레이너를 스레드에서 실행, 콜백은 스레드세이프 ────────
    async def _run_real_training(self, job: Job, trainer) -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        total = max(1, settings.SIM_TOTAL_EPOCHS)

        def log_cb(message: str) -> None:
            loop.call_soon_threadsafe(self._log, job, message)

        def progress_cb(event: dict) -> None:
            loop.call_soon_threadsafe(self._on_progress, job, event)

        result = await asyncio.to_thread(
            trainer.train,
            req=job.request,
            job_id=job.job_id,
            dataset_dir=store.session_dir(job.request.uploadId),
            models_dir=settings.models_dir,
            total_epochs=total,
            log=log_cb,
            progress=progress_cb,
        )
        job.model_path = result.model_path
        return result.final_train_loss, result.final_val_loss

    def _on_progress(self, job: Job, event: dict) -> None:
        epoch = event.get("epoch")
        total = event.get("totalEpochs")
        if isinstance(epoch, int) and isinstance(total, int) and total > 0:
            job.progress = int(epoch / total * 100)
            job.estimated_seconds = 0
        self._emit(job, event)

    # ── 시뮬레이션 학습 (torch 미설치/미지원 태스크 폴백) ─────────────────────
    async def _run_simulation(self, job: Job) -> tuple[float, float]:
        total = max(1, settings.SIM_TOTAL_EPOCHS)
        train_loss = round(random.uniform(0.8, 1.2), 4)
        val_loss = round(train_loss + random.uniform(0.0, 0.15), 4)

        for epoch in range(1, total + 1):
            await asyncio.sleep(settings.SIM_EPOCH_SECONDS)
            train_loss = round(max(0.02, train_loss * random.uniform(0.7, 0.92)), 4)
            val_loss = round(max(0.03, val_loss * random.uniform(0.72, 0.95)), 4)
            job.progress = int(epoch / total * 100)
            job.estimated_seconds = int((total - epoch) * settings.SIM_EPOCH_SECONDS)
            self._log(job, f"Epoch {epoch}/{total} - loss: {train_loss} - val_loss: {val_loss}")
            self._emit(job, {
                "type": "progress",
                "epoch": epoch,
                "totalEpochs": total,
                "trainLoss": train_loss,
                "valLoss": val_loss,
            })

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        out_path = settings.models_dir / f"{job.job_id}_finetuned.pt"
        out_path.write_bytes(
            b"KITECH-FINETUNE-PLACEHOLDER\n"
            + json.dumps({
                "jobId": job.job_id,
                "backboneModelId": job.request.backboneModelId,
                "dataType": job.request.dataType,
                "taskType": job.request.taskType,
                "subTaskType": job.request.subTaskType,
                "finalTrainLoss": train_loss,
                "finalValLoss": val_loss,
            }, ensure_ascii=False).encode("utf-8")
        )
        job.model_path = out_path
        return train_loss, val_loss

    def _log(self, job: Job, message: str) -> None:
        job.logs.append(message)
        self._emit(job, {"type": "log", "message": message})

    def _finish(self, job: Job, event: dict) -> None:
        job._done = True
        self._emit(job, event)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _terminal_event(job: Job) -> str:
    if job.status == "completed":
        return _sse({"type": "complete", "modelDownloadUrl": job.to_response().modelDownloadUrl})
    if job.status == "failed":
        return _sse({"type": "error", "message": job.error or "unknown error"})
    return _sse({"type": "log", "message": "(no active stream)"})


manager = JobManager()
