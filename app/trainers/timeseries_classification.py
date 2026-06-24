"""시계열 분류 트레이너 (sklearn RandomForest).

업로드 디렉터리 구조:
  {dataset_dir}/csv/*.csv   ← 시계열 데이터 (utf-8)
  또는
  {dataset_dir}/json_data/*.json

CSV 형식: timestamp(선택), feature_1, feature_2, ..., label

마지막 열을 label로 자동 인식합니다.
'label', 'class', 'target' 열 이름도 인식합니다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult


_LABEL_COLS = {"label", "class", "target", "y", "category"}


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
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        random.seed(42)
        np.random.seed(42)

        df = _load_tabular(dataset_dir, log)
        if df is None or len(df) < 4:
            raise ValueError(
                "CSV/JSON 파일을 찾을 수 없거나 데이터가 너무 적습니다(최소 4행).\n"
                "csv/ 또는 json_data/ 폴더에 파일을 업로드해 주세요."
            )

        # ── 라벨 열 추출 ────────────────────────────────────────────────
        label_col = _find_label_col(df, _LABEL_COLS)
        df = _drop_timestamp(df)

        if label_col is None:
            raise ValueError(
                "label/class/target 열을 찾을 수 없습니다.\n"
                "마지막 열 또는 'label', 'class', 'target' 열이 필요합니다."
            )

        y_raw = df[label_col].values
        X = df.drop(columns=[label_col]).select_dtypes(include="number").values.astype(float)

        if X.shape[1] == 0:
            raise ValueError("숫자형 feature 열이 없습니다.")

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        nc = len(le.classes_)

        log(f"[data] {len(df)}행, feature {X.shape[1]}개, 클래스 {nc}개 ({', '.join(map(str, le.classes_[:5]))}...)")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if nc < len(y) else None
        )

        # ── RandomForest 학습 (n_estimators를 epoch에 비례해 늘림) ───────
        n_est = max(50, total_epochs * 5)
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)

        log(f"[train] RandomForest(n_estimators={n_est}) 학습 시작...")
        # 점진적 진행률 표시
        step = max(1, n_est // total_epochs)
        for i in range(1, total_epochs + 1):
            partial_est = min(step * i, n_est)
            partial_clf = RandomForestClassifier(n_estimators=partial_est, random_state=42, n_jobs=-1)
            partial_clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, partial_clf.predict(X_train))
            val_acc   = accuracy_score(y_val,   partial_clf.predict(X_val))
            train_loss = round(1 - train_acc, 4)
            val_loss   = round(1 - val_acc,   4)
            log(f"Epoch {i}/{total_epochs} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
            progress({
                "type": "progress",
                "epoch": i, "totalEpochs": total_epochs,
                "trainLoss": train_loss, "valLoss": val_loss, "valAcc": round(val_acc, 4),
            })

        clf.fit(X_train, y_train)
        final_val_acc = accuracy_score(y_val, clf.predict(X_val))

        # ── 저장 ─────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        with open(out_path, "wb") as f:
            pickle.dump({"model": clf, "label_encoder": le, "arch": "random_forest", "jobId": job_id}, f)

        log(f"[done] 학습 완료 — val_acc={final_val_acc:.4f}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=round(1 - accuracy_score(y_train, clf.predict(X_train)), 4),
            final_val_loss=round(1 - final_val_acc, 4),
            extra={"arch": "random_forest_classifier", "nc": nc},
        )


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _load_tabular(dataset_dir: Path, log: LogFn):
    import pandas as pd
    import json as _json

    dfs = []
    for csv_file in sorted((dataset_dir / "csv").glob("*.csv")) if (dataset_dir / "csv").exists() else []:
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
    import pandas as pd
    lower_map = {c.lower(): c for c in df.columns}
    for name in label_col_names:
        if name in lower_map:
            return lower_map[name]
    # 마지막 열 fallback
    return df.columns[-1]


def _drop_timestamp(df):
    import pandas as pd
    ts_hints = {"timestamp", "time", "date", "datetime", "index"}
    drop_cols = [c for c in df.columns if c.lower() in ts_hints]
    # 또는 datetime으로 파싱 가능한 열
    for c in list(df.columns):
        if c not in drop_cols:
            try:
                import pandas as _pd
                _pd.to_datetime(df[c], errors="raise")
                drop_cols.append(c)
            except Exception:
                pass
    return df.drop(columns=drop_cols, errors="ignore")
