"""태스크 → 실제 트레이너 매핑."""
from __future__ import annotations

from ..schemas import DataType, TaskType
from .base import Trainer


def get_trainer(data_type: DataType, task_type: TaskType) -> Trainer | None:
    if data_type == "image":
        if task_type == "classification":
            from .image_classification import ImageClassificationTrainer
            return ImageClassificationTrainer()
        if task_type == "detection":
            from .image_detection import ImageDetectionTrainer
            return ImageDetectionTrainer()
        if task_type == "segmentation":
            from .image_segmentation import ImageSegmentationTrainer
            return ImageSegmentationTrainer()
        if task_type == "anomaly_detection":
            from .image_anomaly import ImageAnomalyTrainer
            return ImageAnomalyTrainer()

    if data_type == "timeseries":
        if task_type == "classification":
            from .timeseries_classification import TimeseriesClassificationTrainer
            return TimeseriesClassificationTrainer()
        if task_type == "anomaly_detection":
            from .timeseries_anomaly import TimeseriesAnomalyTrainer
            return TimeseriesAnomalyTrainer()
        if task_type == "forecasting":
            from .timeseries_forecasting import TimeseriesForecastingTrainer
            return TimeseriesForecastingTrainer()

    if data_type == "json":
        if task_type == "classification":
            from .json_classification import JsonClassificationTrainer
            return JsonClassificationTrainer()
        if task_type == "regression":
            from .json_regression import JsonRegressionTrainer
            return JsonRegressionTrainer()
        if task_type == "anomaly_detection":
            from .json_anomaly import JsonAnomalyTrainer
            return JsonAnomalyTrainer()

    return None
