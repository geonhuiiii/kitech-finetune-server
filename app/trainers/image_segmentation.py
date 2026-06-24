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

        # ── 분할 80/20 ───────────────────────────────────────────────────
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * 0.2))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:] or pairs

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

        bs = min(8, max(2, len(train_pairs)))
        train_loader = DataLoader(_DS(train_pairs), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(_DS(val_pairs),   batch_size=bs, shuffle=False)

        # ── FCN-ResNet50 ─────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = models.segmentation.fcn_resnet50(
                weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT
            )
            log("[init] FCN-ResNet50 사전학습 가중치 로드 완료")
        except Exception as e:
            log(f"[init] 사전학습 가중치 로드 실패({e}) → 무작위 초기화")
            model = models.segmentation.fcn_resnet50(weights=None)

        # classifier head 교체
        model.classifier[4] = nn.Conv2d(512, nc, kernel_size=1)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        final_train_loss = 0.0
        final_val_loss = 0.0

        for epoch in range(1, total_epochs + 1):
            model.train()
            running, n = 0.0, 0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                out = model(imgs)["out"]
                # 출력 크기 → 마스크 크기에 맞게 보간
                if out.shape[-2:] != masks.shape[-2:]:
                    import torch.nn.functional as F
                    out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()
                running += loss.item() * imgs.size(0)
                n += imgs.size(0)
            train_loss = round(running / max(1, n), 4)

            model.eval()
            v_running, v_n = 0.0, 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    out = model(imgs)["out"]
                    if out.shape[-2:] != masks.shape[-2:]:
                        import torch.nn.functional as F
                        out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    v_running += criterion(out, masks).item() * imgs.size(0)
                    v_n += imgs.size(0)
            val_loss = round(v_running / max(1, v_n), 4)
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
