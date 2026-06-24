"""이미지 분류 실제 파인튜닝 트레이너 (torchvision).

데이터 소스: 업로드 세션 디렉터리의 `class_*` 폴더 (DatasetUploadStep 의 role=class_<name>).
  {dataset_dir}/class_crack/img001.jpg
  {dataset_dir}/class_normal/img101.jpg

가벼운 ResNet18 백본을 사용하고, 사전학습 가중치 다운로드가 가능하면 전이학습,
불가하면 무작위 초기화로 학습합니다(오프라인/사내망에서도 동작).
CPU 에서도 빠르게 돌도록 입력 64x64, 소규모 배치로 구성합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
_INPUT_SIZE = 64


class ImageClassificationTrainer:
    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except Exception:
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
        from PIL import Image

        torch.manual_seed(42)
        random.seed(42)

        # ── 1) class_* 폴더에서 (경로, 라벨) 수집 ────────────────────────────
        class_dirs = sorted(
            d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("class_")
        ) if dataset_dir.exists() else []

        if len(class_dirs) < 2:
            raise ValueError(
                f"이미지 분류는 최소 2개 클래스(class_* 폴더)가 필요합니다. "
                f"발견: {len(class_dirs)}개 (경로: {dataset_dir})"
            )

        class_names = [d.name.removeprefix("class_") for d in class_dirs]
        samples: list[tuple[Path, int]] = []
        for label, d in enumerate(class_dirs):
            for p in d.rglob("*"):
                if p.suffix.lower() in _IMG_EXTS and p.is_file():
                    samples.append((p, label))

        if len(samples) < 4:
            raise ValueError(f"학습 이미지가 너무 적습니다(최소 4장). 발견: {len(samples)}장")

        random.shuffle(samples)
        log(f"[data] 클래스 {len(class_names)}개({', '.join(class_names)}), 총 {len(samples)}장")

        # ── 2) Train/Val 분할 (80/20, 최소 1장씩) ───────────────────────────
        n_val = max(1, int(len(samples) * 0.2))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:] or samples
        log(f"[data] train {len(train_samples)} / val {len(val_samples)}")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tf = transforms.Compose([
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        class _ImageDataset(Dataset):
            def __init__(self, items: list[tuple[Path, int]]):
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int):
                path, label = self.items[idx]
                img = Image.open(path).convert("RGB")
                return tf(img), label

        batch_size = min(16, max(2, len(train_samples)))
        train_loader = DataLoader(_ImageDataset(train_samples), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(_ImageDataset(val_samples), batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"[init] device={device.type}, backbone=resnet18, epochs={total_epochs}")

        # ── 3) 모델 구성 (사전학습 시도 → 실패 시 무작위 초기화) ─────────────
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            log("[init] ImageNet 사전학습 가중치 로드 → 전이학습")
        except Exception as exc:  # 다운로드 실패/오프라인
            model = models.resnet18(weights=None)
            log(f"[init] 사전학습 가중치 사용 불가({exc}) → 무작위 초기화 학습")

        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        final_train_loss = 0.0
        final_val_loss = 0.0

        # ── 4) 학습 루프 ────────────────────────────────────────────────────
        for epoch in range(1, total_epochs + 1):
            model.train()
            running = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
                n += xb.size(0)
            train_loss = round(running / max(1, n), 4)

            # 검증
            model.eval()
            v_running = 0.0
            v_n = 0
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    v_running += loss.item() * xb.size(0)
                    v_n += xb.size(0)
                    correct += (out.argmax(1) == yb).sum().item()
            val_loss = round(v_running / max(1, v_n), 4)
            val_acc = round(correct / max(1, v_n), 4)

            final_train_loss, final_val_loss = train_loss, val_loss
            log(f"Epoch {epoch}/{total_epochs} - loss: {train_loss} - val_loss: {val_loss} - val_acc: {val_acc}")
            progress({
                "type": "progress",
                "epoch": epoch,
                "totalEpochs": total_epochs,
                "trainLoss": train_loss,
                "valLoss": val_loss,
                "valAcc": val_acc,
            })

        # ── 5) 산출물 저장 ──────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "arch": "resnet18",
                "input_size": _INPUT_SIZE,
                "normalize": {"mean": mean, "std": std},
                "jobId": job_id,
                "backboneModelId": req.backboneModelId,
            },
            out_path,
        )
        log("[done] 실제 학습 완료 — state_dict 저장됨")
        return TrainResult(
            model_path=out_path,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            extra={"class_names": class_names},
        )
