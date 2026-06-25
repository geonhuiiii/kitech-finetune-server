"""이미지 분류 실제 파인튜닝 트레이너 (timm 기반).

데이터 분할 전략:
  - 기본: 클래스별 Stratified 80/20 분할
  - 소량 데이터(클래스당 < 10장 또는 전체 < 50장): Stratified K-Fold CV
    각 Fold에서 학습 후 최종 모델은 전체 데이터로 재학습
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

_BACKBONE_MAP: dict[str, tuple[str, int]] = {
    "convnext_v2_b":    ("convnext_base.clip_laion2b_augreg_ft_in1k", 224),
    "efficientnet_v2_m": ("tf_efficientnetv2_m.in21k_ft_in1k",        224),
    "swin_v2_b":        ("swinv2_base_window8_256.ms_in1k",           256),
}
_FALLBACK_BACKBONE = ("resnet18", 64)

# 소량 데이터 판정 기준
_SMALL_MIN_PER_CLASS = 10   # 클래스당 이 장수 미만이면 K-Fold
_SMALL_TOTAL         = 50   # 전체 이 장수 미만이면 K-Fold


# ── 분할 유틸 ─────────────────────────────────────────────────────────────────

def _stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """각 클래스에서 val_ratio 비율을 균등 추출 → (train, val)."""
    rng = random.Random(seed)
    per_class: dict[int, list] = defaultdict(list)
    for item in samples:
        per_class[item[1]].append(item)

    train: list = []
    val:   list = []
    for items in per_class.values():
        rng.shuffle(items)
        n_v = max(1, round(len(items) * val_ratio))
        val.extend(items[:n_v])
        # 클래스당 최소 1장은 train에 보장
        train.extend(items[n_v:] if len(items) > n_v else items)

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _kfold_splits(
    samples: list[tuple[Path, int]],
    n_folds: int,
    seed: int = 42,
) -> list[tuple[list, list]]:
    """Stratified K-Fold 분할 → [(train, val), ...]."""
    rng = random.Random(seed)
    per_class: dict[int, list] = defaultdict(list)
    for item in samples:
        per_class[item[1]].append(item)

    fold_bins: list[list] = [[] for _ in range(n_folds)]
    for items in per_class.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        for i, item in enumerate(shuffled):
            fold_bins[i % n_folds].append(item)

    splits = []
    for k in range(n_folds):
        val = fold_bins[k][:]
        train = [x for i, b in enumerate(fold_bins) for x in b if i != k]
        rng.shuffle(train)
        splits.append((train, val))
    return splits


# ── 트레이너 ──────────────────────────────────────────────────────────────────

class ImageClassificationTrainer:
    def is_available(self) -> bool:
        try:
            import torch       # noqa: F401
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
        from torchvision import transforms
        from PIL import Image as PILImage

        torch.manual_seed(42)
        random.seed(42)

        # ── 1) 백본 결정 ─────────────────────────────────────────────────────
        backbone_id = req.backboneModelId
        timm_name, input_size = _BACKBONE_MAP.get(backbone_id, _FALLBACK_BACKBONE)

        try:
            import timm as _timm  # noqa: F401
            use_timm = True
        except ImportError:
            use_timm = False

        log(f"[init] backbone={timm_name}, input={input_size}px, timm={use_timm}")

        # ── 2) class_* 폴더에서 샘플 수집 ───────────────────────────────────
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
        log(f"[data] 클래스 {len(class_names)}개 ({', '.join(class_names)}), 총 {len(samples)}장")

        # ── 3) 소량 여부 판단 ────────────────────────────────────────────────
        per_class_cnt: dict[int, int] = defaultdict(int)
        for _, lbl in samples:
            per_class_cnt[lbl] += 1
        min_per_class = min(per_class_cnt.values())

        is_small = min_per_class < _SMALL_MIN_PER_CLASS or len(samples) < _SMALL_TOTAL
        n_folds  = max(2, min(5, min_per_class)) if is_small else 1

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
                return tf(PILImage.open(path).convert("RGB")), label

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_cls     = len(class_names)
        criterion = nn.CrossEntropyLoss()
        batch_size = min(16, max(2, len(samples)))

        # ── 4-A) K-Fold CV (소량 데이터) ────────────────────────────────────
        if is_small:
            log(f"[data] 소량 데이터 감지(클래스당 최소 {min_per_class}장) → "
                f"{n_folds}-Fold Stratified CV")

            splits = _kfold_splits(samples, n_folds)
            epochs_per_fold = max(1, total_epochs // n_folds)
            fold_val_accs: list[float] = []

            for fold_idx, (train_s, val_s) in enumerate(splits):
                fold_model = _load_model(timm_name, n_cls, use_timm, log if fold_idx == 0 else lambda _: None)
                fold_model = fold_model.to(device)
                fold_opt = torch.optim.Adam(fold_model.parameters(), lr=1e-3)

                train_loader = DataLoader(_DS(train_s), batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(_DS(val_s),   batch_size=batch_size, shuffle=False)

                log(f"[fold {fold_idx+1}/{n_folds}] train={len(train_s)}, val={len(val_s)}")

                best_fold_acc = 0.0
                for ep in range(1, epochs_per_fold + 1):
                    fold_model.train()
                    r, n = 0.0, 0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        fold_opt.zero_grad()
                        loss = criterion(fold_model(xb), yb)
                        loss.backward()
                        fold_opt.step()
                        r += loss.item() * xb.size(0)
                        n += xb.size(0)
                    t_loss = round(r / max(1, n), 4)

                    fold_model.eval()
                    vr, vn, correct = 0.0, 0, 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            out = fold_model(xb)
                            vr += criterion(out, yb).item() * xb.size(0)
                            vn += xb.size(0)
                            correct += (out.argmax(1) == yb).sum().item()
                    v_loss = round(vr / max(1, vn), 4)
                    v_acc  = round(correct / max(1, vn), 4)
                    best_fold_acc = max(best_fold_acc, v_acc)

                    global_ep = fold_idx * epochs_per_fold + ep
                    log(f"[fold {fold_idx+1}] Epoch {ep}/{epochs_per_fold} "
                        f"- loss: {t_loss} - val_loss: {v_loss} - val_acc: {v_acc}")
                    progress({
                        "type": "progress",
                        "epoch": global_ep,
                        "totalEpochs": n_folds * epochs_per_fold,
                        "trainLoss": t_loss, "valLoss": v_loss, "valAcc": v_acc,
                    })

                fold_val_accs.append(best_fold_acc)
                log(f"[fold {fold_idx+1}] best_val_acc={best_fold_acc:.4f}")

            mean_acc = round(sum(fold_val_accs) / len(fold_val_accs), 4)
            log(f"[CV] {n_folds}-Fold 평균 val_acc={mean_acc:.4f} | fold별: {fold_val_accs}")

            # 최종 모델: 전체 데이터로 재학습
            log("[final] 전체 데이터로 최종 모델 학습 중...")
            model = _load_model(timm_name, n_cls, use_timm, lambda _: None)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            all_loader = DataLoader(_DS(samples), batch_size=batch_size, shuffle=True)

            final_train_loss = 0.0
            for ep in range(1, total_epochs + 1):
                model.train()
                r, n = 0.0, 0
                for xb, yb in all_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    r += loss.item() * xb.size(0)
                    n += xb.size(0)
                final_train_loss = round(r / max(1, n), 4)
                progress({
                    "type": "progress",
                    "epoch": n_folds * epochs_per_fold + ep,
                    "totalEpochs": n_folds * epochs_per_fold + total_epochs,
                    "trainLoss": final_train_loss,
                    "valLoss": round(1 - mean_acc, 4),
                    "valAcc": mean_acc,
                })

            final_val_loss = round(1 - mean_acc, 4)

        # ── 4-B) 일반 Stratified 80/20 ──────────────────────────────────────
        else:
            train_samples, val_samples = _stratified_split(samples)
            log(f"[data] Stratified split → train {len(train_samples)} / val {len(val_samples)}")

            # val 클래스 분포 확인
            val_classes = set(lbl for _, lbl in val_samples)
            if len(val_classes) < n_cls:
                log(f"[warn] val에 {len(val_classes)}/{n_cls}개 클래스만 포함됨 — 데이터 추가를 권장합니다")

            train_loader = DataLoader(_DS(train_samples), batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(_DS(val_samples),   batch_size=batch_size, shuffle=False)

            model = _load_model(timm_name, n_cls, use_timm, log)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            final_train_loss = 0.0
            final_val_loss   = 0.0

            for epoch in range(1, total_epochs + 1):
                model.train()
                r, n = 0.0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    r += loss.item() * xb.size(0)
                    n += xb.size(0)
                train_loss = round(r / max(1, n), 4)

                model.eval()
                vr, vn, correct = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        vr += criterion(out, yb).item() * xb.size(0)
                        vn += xb.size(0)
                        correct += (out.argmax(1) == yb).sum().item()
                val_loss = round(vr / max(1, vn), 4)
                val_acc  = round(correct / max(1, vn), 4)
                final_train_loss, final_val_loss = train_loss, val_loss

                log(f"Epoch {epoch}/{total_epochs} - loss: {train_loss} - val_loss: {val_loss} - val_acc: {val_acc}")
                progress({
                    "type": "progress",
                    "epoch": epoch, "totalEpochs": total_epochs,
                    "trainLoss": train_loss, "valLoss": val_loss, "valAcc": val_acc,
                })

        # ── 5) 저장 ──────────────────────────────────────────────────────────
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
            extra={"class_names": class_names, "arch": timm_name, "n_folds": n_folds},
        )


def _load_model(timm_name: str, n_cls: int, use_timm: bool, log: LogFn):
    import torch.nn as nn
    if use_timm:
        import timm
        try:
            model = timm.create_model(timm_name, pretrained=True, num_classes=n_cls)
            log(f"[init] timm '{timm_name}' 사전학습 가중치 로드 완료 (전이학습)")
            return model
        except Exception as e:
            log(f"[init] 사전학습 실패({e}) → 무작위 초기화")
            try:
                return timm.create_model(timm_name, pretrained=False, num_classes=n_cls)
            except Exception as e2:
                log(f"[init] timm 모델 생성 실패({e2}) → resnet18 폴백")

    from torchvision import models
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        log("[init] resnet18 사전학습 가중치 로드 (폴백)")
    except Exception:
        m = models.resnet18(weights=None)
        log("[init] resnet18 무작위 초기화 (폴백)")
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m
