"""이미지 이상탐지 트레이너 (PatchCore 간이 구현).

WideResNet50-2 백본 layer3 특성을 추출해 메모리 뱅크를 구축합니다.
추론 시 최근접 이웃 거리를 이상 점수로 사용합니다.

업로드 디렉터리 구조:
  {dataset_dir}/normal/ ← 정상(양품) 이미지 (jpg·png)
  {dataset_dir}/anomaly/ ← (선택) 불량 이미지 — 학습 불필요, 검증 전용
"""
from __future__ import annotations

import random
from pathlib import Path

from ..schemas import JobRequest
from .base import LogFn, ProgressFn, TrainResult

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageAnomalyTrainer:
    def is_available(self) -> bool:
        try:
            import torch       # noqa: F401
            import torchvision  # noqa: F401
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
        import numpy as np
        from torchvision import models, transforms
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image as PILImage

        torch.manual_seed(42)
        random.seed(42)

        normal_dir = dataset_dir / "normal"
        anomaly_dir = dataset_dir / "anomaly"

        if not normal_dir.exists() or not any(
            f for f in normal_dir.iterdir() if f.suffix.lower() in _IMG_EXTS
        ):
            raise ValueError(
                "normal/ 폴더가 비어 있거나 없습니다.\n"
                "정상(양품) 이미지를 normal/ 폴더에 업로드해 주세요."
            )

        normal_imgs = sorted(f for f in normal_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
        if len(normal_imgs) < 4:
            raise ValueError(f"정상 이미지가 너무 적습니다(최소 4장). 발견: {len(normal_imgs)}장")

        anom_imgs = (
            sorted(f for f in anomaly_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
            if anomaly_dir.exists()
            else []
        )

        log(f"[data] 정상 {len(normal_imgs)}장, 이상 {len(anom_imgs)}장(검증용)")

        input_size = 224
        tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class _DS(Dataset):
            def __init__(self, paths: list[Path]):
                self.paths = paths
            def __len__(self) -> int:
                return len(self.paths)
            def __getitem__(self, idx: int):
                return tf(PILImage.open(self.paths[idx]).convert("RGB"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── WideResNet50-2 특성 추출기 (layer3만 사용, dim=1024) ─────────────
        try:
            backbone = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            )
            log("[init] WideResNet50-2 사전학습 가중치 로드 완료")
        except Exception as e:
            log(f"[init] 사전학습 실패({e}) → 무작위 초기화")
            backbone = models.wide_resnet50_2(weights=None)

        backbone = backbone.to(device)
        backbone.eval()

        feat_buf: list[torch.Tensor] = []

        def _hook(_, __, output: torch.Tensor) -> None:
            # GAP → (batch, C)
            feat_buf.append(output.mean(dim=[2, 3]).detach().cpu())

        handle = backbone.layer3.register_forward_hook(_hook)

        # ── 정상 이미지 → 메모리 뱅크 구축 ─────────────────────────────────
        log("[train] 정상 이미지 특성 추출 중...")
        loader = DataLoader(_DS(normal_imgs), batch_size=min(16, len(normal_imgs)), shuffle=False)
        all_feats: list[np.ndarray] = []

        for step_i, batch in enumerate(loader, 1):
            feat_buf.clear()
            with torch.no_grad():
                backbone(batch.to(device))
            if feat_buf:
                all_feats.append(feat_buf[0].numpy())  # (bs, 1024)

            pct = int(step_i / len(loader) * 100)
            log(f"[train] 특성 추출 중... {pct}%")
            progress({
                "type": "progress",
                "epoch": step_i,
                "totalEpochs": len(loader),
                "trainLoss": 0.0,
                "valLoss": 0.0,
            })

        handle.remove()

        if not all_feats:
            raise RuntimeError("특성 추출 실패: 특성 벡터가 없습니다.")

        memory_bank = np.concatenate(all_feats, axis=0)  # (N, 1024)
        log(f"[train] 메모리 뱅크 구성 완료: {memory_bank.shape}")

        # ── 검증 (이상 이미지 있으면 AUROC 계산) ────────────────────────────
        val_loss = 0.0
        if anom_imgs:
            handle2 = backbone.layer3.register_forward_hook(_hook)

            def _score_batch(paths: list[Path]) -> list[float]:
                scores: list[float] = []
                ldr = DataLoader(_DS(paths), batch_size=min(16, len(paths)), shuffle=False)
                for batch in ldr:
                    feat_buf.clear()
                    with torch.no_grad():
                        backbone(batch.to(device))
                    if feat_buf:
                        q = feat_buf[0].numpy()  # (bs, 1024)
                        # 최근접 이웃 거리
                        for vec in q:
                            dists = np.linalg.norm(memory_bank - vec, axis=1)
                            scores.append(float(np.min(dists)))
                return scores

            try:
                from sklearn.metrics import roc_auc_score
                norm_scores = _score_batch(normal_imgs)
                anom_scores = _score_batch(anom_imgs)
                labels = [0] * len(norm_scores) + [1] * len(anom_scores)
                all_scores = norm_scores + anom_scores
                if len(set(labels)) == 2:
                    auroc = roc_auc_score(labels, all_scores)
                    val_loss = round(1.0 - auroc, 4)
                    log(f"[eval] AUROC: {auroc:.4f}")
            except Exception as e:
                log(f"[warn] 검증 실패: {e}")
            finally:
                handle2.remove()

        # ── 저장 ─────────────────────────────────────────────────────────────
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{job_id}_finetuned.pt"
        torch.save(
            {
                "memory_bank": torch.tensor(memory_bank, dtype=torch.float32),
                "arch": "patchcore_wide_resnet50_2",
                "input_size": input_size,
                "jobId": job_id,
            },
            out_path,
        )
        log(f"[done] 메모리 뱅크 저장 완료: {out_path.name}")

        progress({
            "type": "progress",
            "epoch": total_epochs,
            "totalEpochs": total_epochs,
            "trainLoss": 0.0,
            "valLoss": val_loss,
        })
        return TrainResult(
            model_path=out_path,
            final_train_loss=0.0,
            final_val_loss=val_loss,
            extra={"arch": "patchcore_wide_resnet50_2", "n_memory": len(memory_bank)},
        )
