"""JSON/표 분류 트레이너 (sklearn RandomForestClassifier).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← CSV 데이터
  또는
  {dataset_dir}/json_data/*.json  ← JSON records

마지막 열 또는 'label', 'class', 'target' 열을 라벨로 인식합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult
from .timeseries_classification import (
    _load_tabular,
    _drop_timestamp,
    _find_label_col,
    _LABEL_COLS,
)


class JsonClassificationTrainer:
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
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # XGBoost 우선 사용, 미설치 시 RandomForest 폴백
        try:
            import xgboost as xgb
            _USE_XGB = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier as _FallbackClf
            _USE_XGB = False

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 4:
            raise ValueError(
                "CSV/JSON 파일이 없거나 데이터가 너무 적습니다(최소 4행).\n"
                "csv/ 또는 json_data/ 폴더에 파일을 업로드해 주세요."
            )

        label_col = _find_label_col(df, _LABEL_COLS)
        df = _drop_timestamp(df)
        if label_col is None:
            raise ValueError("label/class/target 열을 찾을 수 없습니다.")

        y_raw = df[label_col].values
        # 범주형 feature 원-핫 인코딩
        feat_df = df.drop(columns=[label_col])
        feat_df = _encode_categoricals(feat_df)
        X = feat_df.values.astype(float)

        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        nc = len(le.classes_)

        log(f"[data] {len(X)}행, feature {X.shape[1]}개, 클래스 {nc}개")

        strat = y if nc < len(y) else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        n_est = max(50, total_epochs * 5)

        if _USE_XGB:
            log(f"[train] XGBoost(n_estimators={n_est}) 학습 시작...")
            clf = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            log(f"[train] XGBoost 미설치 → RandomForest(n_estimators={n_est}) 사용")
            clf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)

        # 점진적 학습 진행률 시뮬레이션 (XGBoost는 전체 학습 후 결과)
        step = max(1, n_est // total_epochs)
        for i in range(1, total_epochs + 1):
            n = min(step * i, n_est)
            if _USE_XGB:
                partial = xgb.XGBClassifier(
                    n_estimators=n, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, n_jobs=-1,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                partial = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
            partial.fit(X_train_s, y_train)
            train_acc = accuracy_score(y_train, partial.predict(X_train_s))
            val_acc   = accuracy_score(y_val,   partial.predict(X_val_s))
            log(f"Epoch {i}/{total_epochs} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
            progress({
                "type": "progress",
                "epoch": i, "totalEpochs": total_epochs,
                "trainLoss": round(1 - train_acc, 4),
                "valLoss": round(1 - val_acc, 4),
                "valAcc": round(val_acc, 4),
            })

        clf.fit(X_train_s, y_train)
        final_val_acc = accuracy_score(y_val, clf.predict(X_val_s))
        final_train_acc = accuracy_score(y_train, clf.predict(X_train_s))

        arch = "xgboost_classifier" if _USE_XGB else "random_forest_classifier"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": clf, "scaler": scaler, "label_encoder": le,
                "arch": arch, "jobId": job_id,
            }, f)

        log(f"[done] 완료({arch}) — val_acc={final_val_acc:.4f}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=round(1 - final_train_acc, 4),
            final_val_loss=round(1 - final_val_acc, 4),
            extra={"arch": arch, "nc": nc},
        )


def _encode_categoricals(df):
    """범주형 열을 원-핫 인코딩."""
    import pandas as pd
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.select_dtypes(include="number").fillna(0)
