"""JSON/표 분류 트레이너 (XGBoost / RandomForest + Stratified K-Fold).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   <- CSV 데이터
  또는
  {dataset_dir}/json_data/*.json  <- JSON records

마지막 열 또는 'label', 'class', 'target' 열을 라벨로 인식합니다.

데이터 분할 전략:
  - 기본: Stratified 80/20 (StratifiedShuffleSplit)
  - 소량(< 200행 또는 클래스당 < 20행): Stratified K-Fold CV
    각 Fold 평균 val_acc -> 최종 모델은 전체 데이터로 재학습
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

_SMALL_TOTAL       = 200
_SMALL_MIN_PER_CLS = 20


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
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
        from sklearn.metrics import accuracy_score

        try:
            import xgboost as xgb
            _USE_XGB = True
        except ImportError:
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

        y_raw   = df[label_col].values
        feat_df = df.drop(columns=[label_col])
        feat_df = _encode_categoricals(feat_df)
        X       = feat_df.values.astype(float)

        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)
        nc = len(le.classes_)

        cls_counts  = np.bincount(y)
        min_per_cls = int(cls_counts.min())
        is_small    = len(X) < _SMALL_TOTAL or min_per_cls < _SMALL_MIN_PER_CLS
        n_folds     = max(2, min(5, min_per_cls)) if is_small else 1

        log(f"[data] {len(X)}행, feature {X.shape[1]}개, 클래스 {nc}개")

        arch    = "xgboost_classifier" if _USE_XGB else "random_forest_classifier"
        n_est   = max(50, total_epochs * 5)
        step    = max(1, n_est // total_epochs)

        scaler = StandardScaler()

        def _make_clf(n: int):
            if _USE_XGB:
                return xgb.XGBClassifier(
                    n_estimators=n, max_depth=6, learning_rate=0.1,
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)

        # ── K-Fold CV (소량 데이터) ───────────────────────────────────────
        if is_small:
            log(
                f"[data] 소량 데이터(전체 {len(X)}행, 클래스당 최소 {min_per_cls}행) -> "
                f"{n_folds}-Fold Stratified CV ({arch})"
            )
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_val_accs: list[float] = []
            total_steps = total_epochs * n_folds
            global_step = 0

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
                X_tr = scaler.fit_transform(X[tr_idx]) if fold_idx == 0 else scaler.transform(X[tr_idx])
                X_va = scaler.transform(X[va_idx])
                y_tr, y_va = y[tr_idx], y[va_idx]
                log(f"[fold {fold_idx+1}/{n_folds}] train={len(X_tr)}, val={len(X_va)}")

                best_fold_acc = 0.0
                for i in range(1, total_epochs + 1):
                    n = min(step * i, n_est)
                    clf_fold = _make_clf(n)
                    clf_fold.fit(X_tr, y_tr)
                    t_acc = accuracy_score(y_tr, clf_fold.predict(X_tr))
                    v_acc = accuracy_score(y_va, clf_fold.predict(X_va))
                    best_fold_acc = max(best_fold_acc, v_acc)
                    global_step += 1

                    log(
                        f"[fold {fold_idx+1}] Epoch {i}/{total_epochs} "
                        f"- acc: {t_acc:.4f} - val_acc: {v_acc:.4f}"
                    )
                    progress({
                        "type": "progress",
                        "epoch": global_step, "totalEpochs": total_steps,
                        "trainLoss": round(1 - t_acc, 4),
                        "valLoss":   round(1 - v_acc, 4),
                        "valAcc":    round(v_acc, 4),
                    })
                fold_val_accs.append(best_fold_acc)
                log(f"[fold {fold_idx+1}] best_val_acc={best_fold_acc:.4f}")

            mean_acc = round(float(np.mean(fold_val_accs)), 4)
            log(f"[CV] {n_folds}-Fold 평균 val_acc={mean_acc:.4f} | fold별: {fold_val_accs}")

            log("[final] 전체 데이터로 최종 모델 학습 중...")
            X_all = scaler.fit_transform(X)
            clf   = _make_clf(n_est)
            clf.fit(X_all, y)
            final_train_acc = accuracy_score(y, clf.predict(X_all))
            final_val_loss  = round(1 - mean_acc, 4)

        # ── Stratified 80/20 (일반) ───────────────────────────────────────
        else:
            stratify = y if min_per_cls >= 2 else None
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, va_idx = next(
                sss.split(X, y) if stratify is not None
                else iter([(
                    np.arange(int(len(X) * 0.8)),
                    np.arange(int(len(X) * 0.8), len(X)),
                )])
            )
            X_train = scaler.fit_transform(X[tr_idx])
            X_val   = scaler.transform(X[va_idx])
            y_train, y_val = y[tr_idx], y[va_idx]

            val_cls = np.unique(y_val)
            if len(val_cls) < nc:
                log(f"[warn] val에 {len(val_cls)}/{nc}개 클래스만 포함 - 데이터 추가를 권장합니다")

            log(
                f"[data] Stratified split -> train {len(X_train)} / val {len(X_val)} "
                f"({arch})"
            )
            best_val_acc_so_far = -1.0
            best_clf            = None

            for i in range(1, total_epochs + 1):
                n = min(step * i, n_est)
                clf = _make_clf(n)
                clf.fit(X_train, y_train)
                t_acc = accuracy_score(y_train, clf.predict(X_train))
                v_acc = accuracy_score(y_val,   clf.predict(X_val))
                if v_acc > best_val_acc_so_far:
                    best_val_acc_so_far = v_acc
                    best_clf            = clf
                log(f"Epoch {i}/{total_epochs} - acc: {t_acc:.4f} - val_acc: {v_acc:.4f}"
                    + (" ★" if v_acc == best_val_acc_so_far else ""))
                progress({
                    "type": "progress",
                    "epoch": i, "totalEpochs": total_epochs,
                    "trainLoss": round(1 - t_acc, 4),
                    "valLoss":   round(1 - v_acc, 4),
                    "valAcc":    round(v_acc, 4),
                })

            clf             = best_clf
            final_train_acc = accuracy_score(y_train, clf.predict(X_train))
            final_val_loss  = round(1 - best_val_acc_so_far, 4)
            log(f"[best] val_acc={best_val_acc_so_far:.4f} 모델 선택")

        # ── 저장 ─────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": clf, "scaler": scaler, "label_encoder": le,
                "arch": arch, "jobId": job_id, "n_folds": n_folds,
            }, f)

        log(f"[done] 완료({arch}) - val_loss={final_val_loss:.4f}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=round(1 - final_train_acc, 4),
            final_val_loss=final_val_loss,
            extra={"arch": arch, "nc": nc, "n_folds": n_folds},
        )


def _encode_categoricals(df):
    """범주형 열을 원-핫 인코딩."""
    import pandas as pd
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.select_dtypes(include="number").fillna(0)
