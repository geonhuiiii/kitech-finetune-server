"""시계열 예측 트레이너 (선형 추세 + 계절성 분해).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← 시계열 데이터 (utf-8)

CSV 형식: timestamp(선택), target 열 (숫자)
target 외 feature 열이 있으면 다변량 회귀로 처리합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult
from .timeseries_classification import _load_tabular, _drop_timestamp


_TARGET_COLS = {"target", "value", "y", "sales", "demand", "price", "count"}


class TimeseriesForecastingTrainer:
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
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 8:
            raise ValueError("CSV/JSON 파일이 없거나 데이터가 너무 적습니다(최소 8행).")

        # timestamp 열 추출 (있으면 인덱스로)
        df = _drop_timestamp(df)
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] == 0:
            raise ValueError("숫자형 열이 없습니다.")

        # target 열 찾기
        lower_map = {c.lower(): c for c in num_df.columns}
        target_col = None
        for tc in _TARGET_COLS:
            if tc in lower_map:
                target_col = lower_map[tc]
                break
        if target_col is None:
            target_col = num_df.columns[-1]

        y = num_df[target_col].ffill().bfill().values.astype(float)
        feat_cols = [c for c in num_df.columns if c != target_col]

        # ── 슬라이딩 윈도우 특성 생성 ────────────────────────────────────
        window = min(10, max(3, len(y) // 20))
        log(f"[data] {len(y)}행, target='{target_col}', window={window}")

        X_rows, y_rows = [], []
        extra_feats = num_df[feat_cols].values.astype(float) if feat_cols else None

        for i in range(window, len(y)):
            row = list(y[i - window:i])  # 과거 window개 값
            if extra_feats is not None:
                row += list(extra_feats[i])
            # 추가 특성: 인덱스(추세), 주기성
            row += [i, i % 7, i % 12, i % 24]
            X_rows.append(row)
            y_rows.append(y[i])

        X = np.array(X_rows)
        Y = np.array(y_rows)

        # train/val 분할 (시계열이므로 순서 유지)
        n_val = max(1, int(len(X) * 0.2))
        X_train, X_val = X[:-n_val], X[-n_val:]
        Y_train, Y_val = Y[:-n_val], Y[-n_val:]

        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s   = scaler_x.transform(X_val)

        scaler_y = StandardScaler()
        Y_train_s = scaler_y.fit_transform(Y_train.reshape(-1, 1)).ravel()

        # ── Ridge 회귀 (alpha를 epoch에 따라 감소) ───────────────────────
        best_model = None
        best_val_loss = float("inf")

        for i in range(1, total_epochs + 1):
            alpha = max(0.01, 10.0 * (0.7 ** i))
            model = Ridge(alpha=alpha)
            model.fit(X_train_s, Y_train_s)

            pred_train = scaler_y.inverse_transform(model.predict(X_train_s).reshape(-1, 1)).ravel()
            pred_val   = scaler_y.inverse_transform(model.predict(X_val_s).reshape(-1, 1)).ravel()

            train_loss = round(float(mean_absolute_error(Y_train, pred_train)), 4)
            val_loss   = round(float(mean_absolute_error(Y_val,   pred_val)),   4)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

            log(f"Epoch {i}/{total_epochs} - MAE: {train_loss} - val_MAE: {val_loss}")
            progress({
                "type": "progress",
                "epoch": i, "totalEpochs": total_epochs,
                "trainLoss": train_loss, "valLoss": val_loss,
            })

        # ── 저장 ─────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": best_model,
                "scaler_x": scaler_x,
                "scaler_y": scaler_y,
                "window": window,
                "target_col": target_col,
                "arch": "ridge_regression",
                "jobId": job_id,
            }, f)

        log(f"[done] 학습 완료 — val_MAE={best_val_loss:.4f}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=round(float(mean_absolute_error(
                Y_train, scaler_y.inverse_transform(best_model.predict(X_train_s).reshape(-1, 1)).ravel()
            )), 4) if best_model else 0.0,
            final_val_loss=best_val_loss,
            extra={"arch": "ridge_regression", "window": window, "target_col": target_col},
        )
