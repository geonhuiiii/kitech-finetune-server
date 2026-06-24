"""런타임 설정 — 모든 값은 환경 변수로 덮어쓸 수 있습니다."""
from __future__ import annotations

import os
from pathlib import Path

# .env 파일 자동 로드 (python-dotenv가 없으면 조용히 skip)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class Settings:
    # 업로드/모델 산출물 저장 루트 (Docker 볼륨으로 마운트 권장)
    DATA_DIR: Path = Path(os.environ.get("FINETUNE_DATA_DIR", "./data")).expanduser()

    # 업로드 파일 최대 크기 (기본 5GB — 프론트엔드와 동일)
    MAX_FILE_SIZE: int = _int("FINETUNE_MAX_FILE_SIZE", 5 * 1024 * 1024 * 1024)

    # 업로드 세션 자동 정리 TTL (시간)
    UPLOAD_TTL_HOURS: int = _int("FINETUNE_UPLOAD_TTL_HOURS", 24)

    # 학습 시뮬레이션 파라미터 (실제 학습 파이프라인으로 교체 가능)
    SIM_TOTAL_EPOCHS: int = _int("FINETUNE_SIM_EPOCHS", 10)
    SIM_EPOCH_SECONDS: float = _float("FINETUNE_SIM_EPOCH_SECONDS", 3.0)

    # CORS 허용 출처 (쉼표 구분). 기본은 모두 허용.
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get("FINETUNE_CORS_ORIGINS", "*").split(",")
        if o.strip()
    ] or ["*"]

    # 모델 다운로드 URL 생성을 위한 외부 베이스 URL.
    # 비어 있으면 요청 호스트 기준 상대 경로로 생성합니다.
    PUBLIC_BASE_URL: str = os.environ.get("FINETUNE_PUBLIC_BASE_URL", "").strip()

    @property
    def uploads_dir(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.DATA_DIR / "models"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
