"""프론트엔드(src/types/repositoryFinetuneChatbot.ts, API_SPEC.md)와 1:1 대응하는 스키마."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

DataType = Literal["image", "timeseries", "json"]
TaskType = Literal[
    "classification",
    "segmentation",
    "detection",
    "anomaly_detection",
    "forecasting",
    "regression",
]
JobStatus = Literal["queued", "running", "completed", "failed"]


class BackboneModel(BaseModel):
    id: str
    name: str
    githubUrl: str
    paperUrl: Optional[str] = None
    downloadUrl: str
    downloadNote: Optional[str] = None
    localFile: Optional[bool] = None
    framework: Literal["pytorch", "sklearn", "other"]
    description: str
    installPackages: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    dataType: DataType
    taskType: TaskType


class RecommendResponse(BaseModel):
    models: list[BackboneModel]
    isFallback: bool = False


class UploadResponse(BaseModel):
    uploadId: str
    filename: str
    role: str
    received: int
    total: int
    isComplete: bool


class ColumnMapping(BaseModel):
    timestamp: Optional[str] = None
    label: Optional[str] = None
    target: Optional[list[str]] = None
    covariates: Optional[list[str]] = None
    features: Optional[list[str]] = None


class JobRequest(BaseModel):
    dataType: DataType
    taskType: TaskType
    subTaskType: str
    backboneModelId: str
    uploadId: str
    columnMapping: Optional[ColumnMapping] = None
    classNames: Optional[list[str]] = None
    keypointCount: Optional[int] = None


class JobResponse(BaseModel):
    jobId: str
    status: JobStatus
    createdAt: str
    progress: Optional[int] = None
    estimatedSeconds: Optional[int] = None
    logs: Optional[list[str]] = None
    modelDownloadUrl: Optional[str] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    code: Optional[str] = None
    details: dict = Field(default_factory=dict)
