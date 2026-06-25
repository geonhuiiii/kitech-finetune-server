"""이미지 세그멘테이션 트레이너 (torchvision FCN-ResNet50).

업로드 디렉터리 구조:
  {dataset_dir}/image/    ← 원본 이미지 (jpg·png)
  {dataset_dir}/mask_png/ ← 마스크 (픽셀값 = 클래스 인덱스, PNG)

이미지와 마스크의 파일명(확장자 제외)이 동일해야 합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageSegmentationTrainer:
    def is_available(self) -> bool:
        try:
            import torch          # noqa: F401
            import torchvision    # noqa: F401
            from PIL import Image  # noqa: F401
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
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image as PILImage

        torch.manual_seed(42)
        random.seed(42)

        image_dir = dataset_dir / "image"
        mask_dir = dataset_dir / "mask_png"

        if not image_dir.exists():
            raise ValueError(
                "image/ 폴더가 없습니다. image/*.jpg + mask_png/*.png 구조로 업로드해 주세요."
            )
        if not mask_dir.exists():
            raise ValueError(
                "mask_png/ 폴더가 없습니다. 픽셀값이 클래스 인덱스인 PNG 마스크를 업로드해 주세요."
            )

        images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
        pairs: list[tuple[Path, Path]] = []
        for img in images:
            mask = mask_dir / (img.stem + ".png")
            if mask.exists():
                pairs.append((img, mask))

        if len(pairs) < 2:
            raise ValueError(
                f"이미지-마스크 쌍이 부족합니다(최소 2쌍). 발견: {len(pairs)}쌍\n"
                "image/*.jpg 와 mask_png/*.png 의 파일명(확장자 제외)이 같아야 합니다."
            )

        # ── 클래스 수 추론 ───────────────────────────────────────────────
        import numpy as np
        class_vals: set[int] = set()
        for _, mask_path in pairs[:min(20, len(pairs))]:
            arr = np.array(PILImage.open(mask_path).convert("L"))
            class_vals.update(np.unique(arr).tolist())
        nc = max(int(max(class_vals)) + 1, 2)

        log(f"[data] {len(pairs)}쌍, 클래스 {nc}개")

        # ── 소량 여부 → K-Fold or 80/20 ─────────────────────────────────
        _SMALL = 20
        is_small = len(pairs) < _SMALL
        n_folds  = max(2, min(5, len(pairs))) if is_small else 1

        input_size = 256
        img_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class _DS(Dataset):
            def __init__(self, items: list[tuple[Path, Path]]):
                self.items = items
            def __len__(self) -> int:
                return len(self.items)
            def __getitem__(self, idx: int):
                img_path, mask_path = self.items[idx]
                img = img_tf(PILImage.open(img_path).convert("RGB"))
                mask = PILImage.open(mask_path).convert("L")
                mask = mask.resize((input_size, input_size), PILImage.NEAREST)
                mask_t = torch.as_tensor(np.array(mask), dtype=torch.long).clamp(0, nc - 1)
                return img, mask_t

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()

        def _make_model():
            try:
                m = models.segmentation.fcn_resnet50(
                    weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT
                )
            except Exception:
                m = models.segmentation.fcn_resnet50(weights=None)
            m.classifier[4] = nn.Conv2d(512, nc, kernel_size=1)
            return m.to(device)

        def _run_epoch(model, loader, optimizer=None):
            """단일 epoch 학습(optimizer 있음) 또는 검증(None)."""
            import torch.nn.functional as F
            is_train = optimizer is not None
            model.train() if is_train else model.eval()
            total, cnt = 0.0, 0
            ctx = torch.enable_grad() if is_train else torch.no_grad()
            with ctx:
                for imgs, masks in loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    if is_train:
                        optimizer.zero_grad()
                    out = model(imgs)["out"]
                    if out.shape[-2:] != masks.shape[-2:]:
                        out = F.interpolate(out, size=masks.shape[-2:],
                                            mode="bilinear", align_corners=False)
                    loss = criterion(out, masks)
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    total += loss.item() * imgs.size(0)
                    cnt   += imgs.size(0)
            return round(total / max(1, cnt), 4)

        final_train_loss = 0.0
        final_val_loss   = 0.0

        # ── K-Fold (소량) ────────────────────────────────────────────────
        if is_small:
            log(f"[data] 소량 데이터({len(pairs)}쌍) → {n_folds}-Fold CV")
            shuffled = list(pairs)
            random.shuffle(shuffled)
            fold_bins = [shuffled[i::n_folds] for i in range(n_folds)]

            epochs_per_fold = max(1, total_epochs // n_folds)
            fold_val_losses: list[float] = []

            for fold_idx, val_bin in enumerate(fold_bins):
                train_p = [x for i, b in enumerate(fold_bins) for x in b if i != fold_idx]
                log(f"[fold {fold_idx+1}/{n_folds}] train={len(train_p)}, val={len(val_bin)}")

                bs           = min(8, max(2, len(train_p)))
                train_loader = DataLoader(_DS(train_p),  batch_size=bs, shuffle=True)
                val_loader_f = DataLoader(_DS(val_bin),  batch_size=bs, shuffle=False)

                model     = _make_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                best_val = float("inf")
                for ep in range(1, epochs_per_fold + 1):
                    t_loss = _run_epoch(model, train_loader, optimizer)
                    v_loss = _run_epoch(model, val_loader_f)
                    best_val = min(best_val, v_loss)
                    global_ep = fold_idx * epochs_per_fold + ep
                    log(f"[fold {fold_idx+1}] Epoch {ep}/{epochs_per_fold} "
                        f"- loss: {t_loss} - val_loss: {v_loss}")
                    progress({
                        "type": "progress",
                        "epoch": global_ep,
                        "totalEpochs": n_folds * epochs_per_fold,
                        "trainLoss": t_loss, "valLoss": v_loss,
                    })
                fold_val_losses.append(best_val)

            mean_val = round(sum(fold_val_losses) / len(fold_val_losses), 4)
            log(f"[CV] {n_folds}-Fold 평균 val_loss={mean_val:.4f}")

            # 최종 모델: 전체 데이터로 재학습
            log("[final] 전체 데이터로 최종 모델 학습 중...")
            model     = _make_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            all_loader = DataLoader(_DS(shuffled), batch_size=min(8, len(shuffled)), shuffle=True)
            for ep in range(1, total_epochs + 1):
                t_loss = _run_epoch(model, all_loader, optimizer)
                progress({
                    "type": "progress",
                    "epoch": n_folds * epochs_per_fold + ep,
                    "totalEpochs": n_folds * epochs_per_fold + total_epochs,
                    "trainLoss": t_loss, "valLoss": mean_val,
                })
            final_train_loss = t_loss
            final_val_loss   = mean_val

        # ── 일반 80/20 ───────────────────────────────────────────────────
        else:
            random.shuffle(pairs)
            n_val       = max(1, int(len(pairs) * 0.2))
            val_pairs   = pairs[:n_val]
            train_pairs = pairs[n_val:]
            log(f"[data] 80/20 split -> train {len(train_pairs)} / val {len(val_pairs)}")

            bs           = min(8, max(2, len(train_pairs)))
            train_loader = DataLoader(_DS(train_pairs), batch_size=bs, shuffle=True)
            val_loader   = DataLoader(_DS(val_pairs),   batch_size=bs, shuffle=False)

            model     = _make_model()
            log("[init] FCN-ResNet50 로드 완료")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(1, total_epochs + 1):
                train_loss = _run_epoch(model, train_loader, optimizer)
                val_loss   = _run_epoch(model, val_loader)
                final_train_loss, final_val_loss = train_loss, val_loss

                log(f"Epoch {epoch}/{total_epochs} - loss: {train_loss} - val_loss: {val_loss}")
                progress({
                    "type": "progress",
                    "epoch": epoch, "totalEpochs": total_epochs,
                    "trainLoss": train_loss, "valLoss": val_loss,
                })

        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "nc": nc,
            "input_size": input_size,
            "arch": "fcn_resnet50",
            "jobId": job_id,
        }, out_path)
        log(f"[done] 학습 완료 — 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            extra={"nc": nc, "arch": "fcn_resnet50"},
        )
