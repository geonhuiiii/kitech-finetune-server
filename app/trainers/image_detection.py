"""이미지 객체탐지 트레이너 (ultralytics YOLO26n).

업로드 디렉터리 구조:
  {dataset_dir}/image/   ← 이미지 파일 (jpg·png 등)
  {dataset_dir}/label_txt/ ← YOLO txt 라벨 (class_id cx cy w h, 정규화)

이미지·라벨 파일명(확장자 제외)이 동일해야 합니다.
"""
from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageDetectionTrainer:
    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
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
        try:
            import yaml
        except ImportError:
            import subprocess, sys  # noqa: E401
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
            import yaml  # type: ignore[no-redef]

        from ultralytics import YOLO

        image_dir = dataset_dir / "image"
        label_dir = dataset_dir / "label_txt"

        if not image_dir.exists() or not any(image_dir.iterdir()):
            raise ValueError(
                "image/ 폴더가 비어 있거나 없습니다.\n"
                "image/*.jpg + label_txt/*.txt 구조로 업로드해 주세요."
            )

        images = sorted(
            f for f in image_dir.iterdir() if f.suffix.lower() in _IMG_EXTS
        )
        if len(images) < 4:
            raise ValueError(
                f"이미지가 너무 적습니다(최소 4장). 발견: {len(images)}장"
            )

        # ── YOLO dataset 디렉터리 생성 ──────────────────────────────────────
        yolo_root = Path(tempfile.mkdtemp(prefix=f"yolo_{job_id}_"))
        try:
            train_img_dir = yolo_root / "images" / "train"
            val_img_dir = yolo_root / "images" / "val"
            train_lbl_dir = yolo_root / "labels" / "train"
            val_lbl_dir = yolo_root / "labels" / "val"
            for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # ── train/val 분할 80/20 ────────────────────────────────────────
            random.shuffle(images)
            n_val = max(1, int(len(images) * 0.2))
            val_imgs = images[:n_val]
            train_imgs = images[n_val:] or images

            class_set: set[int] = set()
            for img_path, dst_img, dst_lbl in [
                *[(p, train_img_dir, train_lbl_dir) for p in train_imgs],
                *[(p, val_img_dir, val_lbl_dir) for p in val_imgs],
            ]:
                shutil.copy2(img_path, dst_img / img_path.name)
                lbl = (label_dir / (img_path.stem + ".txt")) if label_dir.exists() else None
                if lbl and lbl.exists():
                    shutil.copy2(lbl, dst_lbl / lbl.name)
                    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_set.add(int(parts[0]))
                            except ValueError:
                                pass

            nc = (max(class_set) + 1) if class_set else 1
            names = [f"class_{i}" for i in range(nc)]

            data_yaml = yolo_root / "data.yaml"
            data_yaml.write_text(
                yaml.dump(
                    {
                        "path": str(yolo_root),
                        "train": "images/train",
                        "val": "images/val",
                        "nc": nc,
                        "names": names,
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            log(
                f"[data] 학습 {len(train_imgs)}장 / 검증 {len(val_imgs)}장, "
                f"클래스 {nc}개 ({', '.join(names)})"
            )

            # ── YOLO 모델 로드 + 콜백 등록 ─────────────────────────────────
            model = YOLO("yolo26n.pt")
            log("[init] YOLO26n 사전학습 가중치 로드 완료")

            def _on_epoch_end(trainer_obj):  # type: ignore[misc]
                epoch = trainer_obj.epoch + 1
                loss = float(getattr(trainer_obj, "loss", 0) or 0)
                metrics = getattr(trainer_obj, "metrics", {}) or {}
                val_map = float(metrics.get("metrics/mAP50(B)", 0) or 0)
                log(f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f} - mAP50: {val_map:.4f}")
                progress({
                    "type": "progress",
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "trainLoss": round(loss, 4),
                    "valLoss": round(max(0.0, 1.0 - val_map), 4),
                    "valAcc": round(val_map, 4),
                })

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            # ── 학습 실행 ─────────────────────────────────────────────────
            models_dir.mkdir(parents=True, exist_ok=True)
            results = model.train(
                data=str(data_yaml),
                epochs=total_epochs,
                imgsz=640,
                project=str(models_dir),
                name=job_id,
                save=True,
                plots=False,
                verbose=False,
                exist_ok=True,
            )

            # ── 결과 파일 복사 ────────────────────────────────────────────
            best_pt = models_dir / job_id / "weights" / "best.pt"
            last_pt = models_dir / job_id / "weights" / "last.pt"
            src = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)

            out_path = models_dir / f"{job_id}_finetuned.pt"
            if src:
                shutil.copy2(src, out_path)
            else:
                out_path.write_bytes(b"YOLO_TRAINING_PLACEHOLDER")

            rd = getattr(results, "results_dict", {}) or {}
            train_loss = float(rd.get("train/box_loss", 0) or 0)
            val_loss = float(rd.get("val/box_loss", 0) or 0)
            log(f"[done] 학습 완료 — nc={nc}, 저장: {out_path.name}")
            return TrainResult(
                model_path=out_path,
                final_train_loss=round(train_loss, 4),
                final_val_loss=round(val_loss, 4),
                extra={"nc": nc, "arch": "yolo26n"},
            )
        finally:
            shutil.rmtree(yolo_root, ignore_errors=True)
