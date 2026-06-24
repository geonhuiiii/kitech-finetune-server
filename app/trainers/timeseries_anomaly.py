"""시계열 이상탐지 트레이너 (IsolationForest).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← 시계열 데이터 (utf-8)

CSV 형식: timestamp(선택), feature_1, feature_2, ...
라벨 열(있으면 제외 후 학습, 검증에 사용).
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult
from .timeseries_classification import _load_tabular, _drop_timestamp, _find_label_col, _LABEL_COLS


class TimeseriesAnomalyTrainer:
    def is_available(self) -> bool:
        try:
            import pandas  # noqa: F401
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

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
        import pickle
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 4:
            raise ValueError("CSV/JSON 파일이 없거나 데이터가 너무 적습니다(최소 4행).")

        # label 열 있으면 제외 후 특성으로 사용
        label_col = _find_label_col(df, _LABEL_COLS)
        y = None
        if label_col and set(df[label_col].dropna().unique()).issubset({0, 1, "0", "1", True, False, "normal", "anomaly"}):
            y = df[label_col].values
            df = df.drop(columns=[label_col])
        else:
            label_col = None

        df = _drop_timestamp(df)
        X = df.select_dtypes(include="number").fillna(0).values.astype(float)
        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        log(f"[data] {len(X)}행, feature {X.shape[1]}개")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── IsolationForest (contamination 자동 추정) ────────────────────
        contamination = 0.1  # 기본 10%
        if y is not None:
            try:
                n_anom = sum(1 for v in y if str(v) in {"1", "anomaly", "True"})
                contamination = max(0.01, min(0.5, n_anom / len(y)))
            except Exception:
                pass

        n_est = max(100, total_epochs * 10)
        log(f"[train] IsolationForest(n_estimators={n_est}, contamination={contamination:.2f}) 학습...")

        clf = IsolationForest(n_estimators=n_est, contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X_scaled)

        # 진행률 표시
        for i in range(1, total_epochs + 1):
            progress({
                "type": "progress",
                "epoch": i, "totalEpochs": total_epochs,
                "trainLoss": 0.0, "valLoss": 0.0,
            })

        # 검증 (라벨 있는 경우)
        val_loss = 0.0
        if y is not None:
            try:
                from sklearn.metrics import roc_auc_score
                scores = -clf.score_samples(X_scaled)
                y_bin = np.array([1 if str(v) in {"1", "anomaly", "True"} else 0 for v in y])
                if len(np.unique(y_bin)) == 2:
                    auroc = roc_auc_score(y_bin, scores)
                    val_loss = round(1 - auroc, 4)
                    log(f"[eval] AUROC: {auroc:.4f}")
            except Exception as e:
                log(f"[warn] 검증 실패: {e}")

        # ── 저장 ─────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": clf, "scaler": scaler,
                "arch": "isolation_forest", "jobId": job_id
            }, f)

        log(f"[done] 학습 완료, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=0.0,
            final_val_loss=val_loss,
            extra={"arch": "isolation_forest", "contamination": contamination},
        )
