"""태스크 → 실제 트레이너 매핑.

지원하지 않는 태스크이거나 ML 의존성이 없으면 None 을 반환하고,
jobs_engine 은 시뮬레이션 학습으로 폴백합니다.
"""
from __future__ import annotations

from ..schemas import DataType, TaskType
from .base import Trainer


def get_trainer(data_type: DataType, task_type: TaskType) -> Trainer | None:
    if data_type == "image" and task_type == "classification":
        from .image_classification import ImageClassificationTrainer

        return ImageClassificationTrainer()
    # TODO: 세그멘테이션/검출/시계열/표 등 실제 트레이너 추가 지점
    return None
