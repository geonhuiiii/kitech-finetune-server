"""이미지 분류 실제 파인튜닝 트레이너 (timm 기반).

선택된 backboneModelId 에 맞는 모델을 timm 으로 로드합니다.
사전학습 가중치 다운로드 실패 시 무작위 초기화로 폴백합니다.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# backboneModelId → (timm model name, input size)
# pretrained=True 시 timm 이 허깅페이스/공식 URL 에서 가중치를 자동 다운로드합니다.
_BACKBONE_MAP: dict[str, tuple[str, int]] = {
    "convnext_v2_b":    ("convnext_base.clip_laion2b_augreg_ft_in1k", 224),
    "efficientnet_v2_m": ("tf_efficientnetv2_m.in21k_ft_in1k",        224),
    "swin_v2_b":        ("swinv2_base_window8_256.ms_in1k",           256),
}
_FALLBACK_BACKBONE = ("resnet18", 64)   # timm 없거나 미등록 모델 폴백


class ImageClassificationTrainer:
    def is_available(self) -> bool:
        try:
            import torch       # noqa: F401
            import torchvision # noqa: F401
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
        from torchvision import transforms
        from PIL import Image as PILImage

        torch.manual_seed(42)
        random.seed(42)

        # ── 1) 백본 결정 ─────────────────────────────────────────────────────
        backbone_id = req.backboneModelId
        timm_name, input_size = _BACKBONE_MAP.get(backbone_id, _FALLBACK_BACKBONE)

        # timm 사용 가능 여부 확인
        try:
            import timm as _timm  # noqa: F401
            use_timm = True
        except ImportError:
            use_timm = False

        log(f"[init] backbone_id={backbone_id}, timm={use_timm}, arch={timm_name}, input={input_size}px")

        # ── 2) class_* 폴더에서 (경로, 라벨) 수집 ───────────────────────────
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

        # ── 3) Train/Val 분할 (80/20) ────────────────────────────────────────
        n_val = max(1, int(len(samples) * 0.2))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:] or samples
        log(f"[data] train {len(train_samples)} / val {len(val_samples)}")

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        class _DS(Dataset):
            def __init__(self, items: list[tuple[Path, int]]):
                self.items = items
            def __len__(self) -> int:
                return len(self.items)
            def __getitem__(self, idx: int):
                path, label = self.items[idx]
                img = PILImage.open(path).convert("RGB")
                return tf(img), label

        batch_size = min(16, max(2, len(train_samples)))
        train_loader = DataLoader(_DS(train_samples), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(_DS(val_samples),   batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_cls  = len(class_names)

        # ── 4) 모델 로드 ─────────────────────────────────────────────────────
        model = _load_model(timm_name, n_cls, use_timm, log)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        final_train_loss = 0.0
        final_val_loss   = 0.0

        # ── 5) 학습 루프 ─────────────────────────────────────────────────────
        for epoch in range(1, total_epochs + 1):
            model.train()
            running, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
                n += xb.size(0)
            train_loss = round(running / max(1, n), 4)

            model.eval()
            v_running, v_n, correct = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    v_running += criterion(out, yb).item() * xb.size(0)
                    v_n += xb.size(0)
                    correct += (out.argmax(1) == yb).sum().item()
            val_loss = round(v_running / max(1, v_n), 4)
            val_acc  = round(correct / max(1, v_n), 4)

            final_train_loss, final_val_loss = train_loss, val_loss
            log(f"Epoch {epoch}/{total_epochs} - loss: {train_loss} - val_loss: {val_loss} - val_acc: {val_acc}")
            progress({
                "type": "progress",
                "epoch": epoch, "totalEpochs": total_epochs,
                "trainLoss": train_loss, "valLoss": val_loss, "valAcc": val_acc,
            })

        # ── 6) 저장 ──────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "arch": timm_name,
            "input_size": input_size,
            "normalize": {"mean": mean, "std": std},
            "jobId": job_id,
            "backboneModelId": backbone_id,
        }, out_path)
        log(f"[done] 학습 완료 — arch={timm_name}, 저장: {out_path.name}")
        return TrainResult(
            model_path=out_path,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            extra={"class_names": class_names, "arch": timm_name},
        )


def _load_model(timm_name: str, n_cls: int, use_timm: bool, log: LogFn):
    """timm 으로 선택된 백본을 로드. 실패 시 torchvision ResNet18 으로 폴백."""
    import torch.nn as nn

    if use_timm:
        import timm
        # 사전학습 가중치 시도
        try:
            model = timm.create_model(timm_name, pretrained=True, num_classes=n_cls)
            log(f"[init] timm '{timm_name}' 사전학습 가중치 로드 완료 (전이학습)")
            return model
        except Exception as e:
            log(f"[init] 사전학습 가중치 다운로드 실패({e}) → 무작위 초기화")
            try:
                model = timm.create_model(timm_name, pretrained=False, num_classes=n_cls)
                log(f"[init] timm '{timm_name}' 무작위 초기화")
                return model
            except Exception as e2:
                log(f"[init] timm 모델 생성 실패({e2}) → resnet18 폴백")

    # torchvision ResNet18 최종 폴백
    from torchvision import models
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        log("[init] resnet18 사전학습 가중치 로드 (폴백)")
    except Exception:
        m = models.resnet18(weights=None)
        log("[init] resnet18 무작위 초기화 (폴백)")
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m
