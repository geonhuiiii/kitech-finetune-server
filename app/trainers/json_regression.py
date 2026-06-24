"""JSON/표 회귀 트레이너 (sklearn RandomForestRegressor).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← CSV 데이터
  또는
  {dataset_dir}/json_data/*.json

마지막 열 또는 'target', 'value', 'y' 열을 예측 대상으로 인식합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult
from .timeseries_classification import _load_tabular, _drop_timestamp, _find_label_col
from .json_classification import _encode_categoricals

_TARGET_COLS = {"target", "value", "y", "output", "price", "score", "amount"}


class JsonRegressionTrainer:
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
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        try:
            import xgboost as xgb
            _USE_XGB = True
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor as _FallbackReg
            _USE_XGB = False

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 4:
            raise ValueError(
                "CSV/JSON 파일이 없거나 데이터가 너무 적습니다(최소 4행)."
            )

        df = _drop_timestamp(df)
        target_col = _find_label_col(df, _TARGET_COLS)
        if target_col is None:
            raise ValueError("target/value/y 열을 찾을 수 없습니다.")

        y = df[target_col].values.astype(float)
        feat_df = _encode_categoricals(df.drop(columns=[target_col]))
        X = feat_df.values.astype(float)

        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        log(f"[data] {len(X)}행, feature {X.shape[1]}개, target='{target_col}'")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        n_est = max(50, total_epochs * 5)
        step = max(1, n_est // total_epochs)
        best_val_loss = float("inf")
        best_model = None

        for i in range(1, total_epochs + 1):
            n = min(step * i, n_est)
            if _USE_XGB:
                partial = xgb.XGBRegressor(
                    n_estimators=n, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1,
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                partial = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
            partial.fit(X_train_s, y_train)
            train_mae = mean_absolute_error(y_train, partial.predict(X_train_s))
            val_mae   = mean_absolute_error(y_val,   partial.predict(X_val_s))
            if val_mae < best_val_loss:
                best_val_loss = val_mae
                best_model = partial
            log(f"Epoch {i}/{total_epochs} - MAE: {train_mae:.4f} - val_MAE: {val_mae:.4f}")
            progress({
                "type": "progress",
                "epoch": i, "totalEpochs": total_epochs,
                "trainLoss": round(float(train_mae), 4),
                "valLoss": round(float(val_mae), 4),
            })

        val_r2 = r2_score(y_val, best_model.predict(X_val_s))
        log(f"[done] val_R²={val_r2:.4f}")

        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        arch = "xgboost_regressor" if _USE_XGB else "random_forest_regressor"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": best_model, "scaler": scaler,
                "target_col": target_col,
                "arch": arch, "jobId": job_id,
            }, f)

        return TrainResult(
            model_path=out_path,
            final_train_loss=round(float(mean_absolute_error(y_train, best_model.predict(X_train_s))), 4),
            final_val_loss=round(best_val_loss, 4),
            extra={"arch": arch, "r2": round(float(val_r2), 4)},
        )
