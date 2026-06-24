"""트레이너 공통 인터페이스.

실제 학습 트레이너는 동기(블로킹) 함수입니다. jobs_engine 이 별도 스레드에서 실행하고,
로그/진행률은 스레드세이프 콜백(log, progress)으로 전달합니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..schemas import JobRequest

LogFn = Callable[[str], None]
ProgressFn = Callable[[dict], None]


@dataclass
class TrainResult:
    model_path: Path
    final_train_loss: float
    final_val_loss: float
    extra: dict = field(default_factory=dict)


class Trainer(Protocol):
    def is_available(self) -> bool:
        """필요한 ML 의존성(torch 등)이 설치돼 있으면 True."""
        ...

    def train(
        self,
        *,
        req: JobRequest,
        job_id: str,
        dataset_dir: Path,
        models_dir: Path,
        total_epochs: int,
        log: LogFn,
        progress: ProgressFn,
    ) -> TrainResult:
        ...
