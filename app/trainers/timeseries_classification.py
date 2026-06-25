"""시계열 분류 트레이너 (sklearn RandomForest + Stratified K-Fold).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← 시계열 데이터 (utf-8)
  또는
  {dataset_dir}/json_data/*.json

CSV 형식: timestamp(선택), feature_1, feature_2, ..., label

마지막 열을 label로 자동 인식합니다.
'label', 'class', 'target' 열 이름도 인식합니다.

데이터 분할 전략:
  - 기본: Stratified 80/20 (sklearn StratifiedShuffleSplit)
  - 소량(< 200행 또는 클래스당 < 20행): Stratified K-Fold CV
    각 Fold에서 평균 val_acc 계산 → 최종 모델은 전체 데이터로 재학습
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult


_LABEL_COLS = {"label", "class", "target", "y", "category"}

# 소량 데이터 판정 기준
_SMALL_TOTAL        = 200   # 전체 행 수
_SMALL_MIN_PER_CLS  = 20    # 클래스당 최소 행 수


class TimeseriesClassificationTrainer:
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
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
        from sklearn.metrics import accuracy_score

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 4:
            raise ValueError(
                "CSV/JSON 파일을 찾을 수 없거나 데이터가 너무 적습니다(최소 4행).\n"
                "csv/ 또는 json_data/ 폴더에 파일을 업로드해 주세요."
            )

        label_col = _find_label_col(df, _LABEL_COLS)
        df = _drop_timestamp(df)

        if label_col is None:
            raise ValueError(
                "label/class/target 열을 찾을 수 없습니다.\n"
                "'label', 'class', 'target' 열 또는 마지막 열이 필요합니다."
            )

        y_raw = df[label_col].values
        X = df.drop(columns=[label_col]).select_dtypes(include="number").values.astype(float)
        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        nc = len(le.classes_)

        log(f"[data] {len(df)}행, feature {X.shape[1]}개, 클래스 {nc}개 "
            f"({', '.join(map(str, le.classes_[:5]))}...)")

        # ── 소량 여부 판단 ────────────────────────────────────────────────
        cls_counts = np.bincount(y)
        min_per_cls = int(cls_counts.min())
        is_small = len(X) < _SMALL_TOTAL or min_per_cls < _SMALL_MIN_PER_CLS
        n_folds  = max(2, min(5, min_per_cls)) if is_small else 1

        n_est = max(50, total_epochs * 5)
        step  = max(1, n_est // total_epochs)

        def _make_clf(n: int) -> RandomForestClassifier:
            return RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)

        # ── K-Fold CV (소량 데이터) ───────────────────────────────────────
        if is_small:
            log(f"[data] 소량 데이터(전체 {len(X)}행, 클래스당 최소 {min_per_cls}행) → "
                f"{n_folds}-Fold Stratified CV")

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_val_accs: list[float] = []
            global_step  = 0
            total_steps  = total_epochs * n_folds

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
                X_tr, X_va = X[tr_idx], X[va_idx]
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

                    log(f"[fold {fold_idx+1}] Epoch {i}/{total_epochs} "
                        f"- acc: {t_acc:.4f} - val_acc: {v_acc:.4f}")
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

            # 최종 모델: 전체 데이터로 재학습
            log("[final] 전체 데이터로 최종 모델 학습 중...")
            clf = _make_clf(n_est)
            clf.fit(X, y)
            final_train_acc = accuracy_score(y, clf.predict(X))
            final_val_loss  = round(1 - mean_acc, 4)
            log(f"[final] train_acc={final_train_acc:.4f}, CV val_acc={mean_acc:.4f}")

        # ── Stratified 80/20 (일반) ───────────────────────────────────────
        else:
            # min per class가 2 미만이면 stratify 불가
            stratify = y if min_per_cls >= 2 else None
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_idx, va_idx = next(sss.split(X, y) if stratify is not None
                                  else iter([(np.arange(int(len(X)*0.8)),
                                              np.arange(int(len(X)*0.8), len(X)))]))
            X_train, X_val = X[tr_idx], X[va_idx]
            y_train, y_val = y[tr_idx], y[va_idx]

            val_cls = np.unique(y_val)
            if len(val_cls) < nc:
                log(f"[warn] val에 {len(val_cls)}/{nc}개 클래스만 포함 — 데이터 추가를 권장합니다")

            log(f"[data] Stratified split → train {len(X_train)} / val {len(X_val)}")

            for i in range(1, total_epochs + 1):
                n = min(step * i, n_est)
                clf = _make_clf(n)
                clf.fit(X_train, y_train)
                t_acc = accuracy_score(y_train, clf.predict(X_train))
                v_acc = accuracy_score(y_val,   clf.predict(X_val))
                log(f"Epoch {i}/{total_epochs} - acc: {t_acc:.4f} - val_acc: {v_acc:.4f}")
                progress({
                    "type": "progress",
                    "epoch": i, "totalEpochs": total_epochs,
                    "trainLoss": round(1 - t_acc, 4),
                    "valLoss":   round(1 - v_acc, 4),
                    "valAcc":    round(v_acc, 4),
                })

            clf = _make_clf(n_est)
            clf.fit(X_train, y_train)
            final_train_acc = accuracy_score(y_train, clf.predict(X_train))
            final_val_loss  = round(1 - accuracy_score(y_val, clf.predict(X_val)), 4)

        # ── 저장 ─────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": clf, "label_encoder": le,
                "arch": "random_forest", "jobId": job_id,
                "n_folds": n_folds,
            }, f)

        log(f"[done] 학습 완료 — val_loss={final_val_loss:.4f}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=round(1 - final_train_acc, 4),
            final_val_loss=final_val_loss,
            extra={"arch": "random_forest_classifier", "nc": nc, "n_folds": n_folds},
        )


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _load_tabular(dataset_dir: Path, log: LogFn):
    import pandas as pd
    import json as _json

    dfs = []
    csv_dir = dataset_dir / "csv"
    if csv_dir.exists():
        for csv_file in sorted(csv_dir.glob("*.csv")):
            try:
                dfs.append(pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip"))
                log(f"[data] 로드: {csv_file.name}")
            except Exception as e:
                log(f"[warn] {csv_file.name} 로드 실패: {e}")

    json_dir = dataset_dir / "json_data"
    if json_dir.exists():
        for jf in sorted(json_dir.glob("*.json")):
            try:
                data = _json.loads(jf.read_text(encoding="utf-8"))
                dfs.append(pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data]))
                log(f"[data] 로드: {jf.name}")
            except Exception as e:
                log(f"[warn] {jf.name} 로드 실패: {e}")

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _find_label_col(df, label_col_names: set[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in label_col_names:
        if name in lower_map:
            return lower_map[name]
    return df.columns[-1]


def _drop_timestamp(df):
    import pandas as _pd
    ts_hints = {"timestamp", "time", "date", "datetime", "index"}
    drop_cols = [c for c in df.columns if c.lower() in ts_hints]
    for c in list(df.columns):
        if c not in drop_cols:
            try:
                _pd.to_datetime(df[c], errors="raise")
                drop_cols.append(c)
            except Exception:
                pass
    return df.drop(columns=drop_cols, errors="ignore")
