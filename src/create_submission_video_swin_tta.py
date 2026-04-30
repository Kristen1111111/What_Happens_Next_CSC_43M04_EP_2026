#!/usr/bin/env python3
"""
Create a Kaggle submission with temporal test-time augmentation (TTA).

Use from src/:
    python create_submission.py experiment=video_swin_transformer_essai \
      training.checkpoint_path=checkpoints/video_swin_t_best.pt

Important:
- imports build_model from train2, because Video Swin is registered there;
- loads the exact Hydra config saved in the checkpoint;
- uses the checkpoint's num_frames / pretrained normalization;
- averages logits over several temporal views;
- writes: video_name,predicted_class
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_unlabeled_video_samples
from train2 import build_model
from utils import build_transforms, set_seed


def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader if row.get("video_name", "").strip()]


def make_submission_path(base_output_path: Path, val_accuracy: Optional[float]) -> Path:
    if base_output_path.is_dir():
        base_output_path = base_output_path / "submission.csv"

    suffix = base_output_path.suffix or ".csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_suffix = f"acc{val_accuracy:.4f}" if val_accuracy is not None else "accNA"
    return base_output_path.with_name(f"{base_output_path.stem}_{acc_suffix}_{timestamp}{suffix}")


def build_model_from_checkpoint(ckpt: Dict[str, Any]) -> torch.nn.Module:
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(
            "Checkpoint has no full Hydra config. Retrain with train2.py so the full config is saved."
        )
    train_cfg = OmegaConf.create(ckpt["config"])
    return build_model(train_cfg)


def load_state_dict_strict_or_clean(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Accept normal and DataParallel checkpoints."""
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        cleaned = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=True)


@torch.no_grad()
def logits_for_temporal_view(
    model: torch.nn.Module,
    samples,
    test_root: Path,
    transform,
    num_frames: int,
    view_idx: int,
    n_views: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=transform,
        sample_list=samples,
        temporal_view_index=view_idx,
        temporal_num_views=n_views,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    all_logits: List[torch.Tensor] = []
    log_interval = max(1, len(loader) // 10)

    for batch_idx, (video_batch, _labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch).detach().float().cpu()
        all_logits.append(logits)

        if batch_idx == len(loader) or batch_idx % log_interval == 0:
            print(f"  TTA view {view_idx + 1}/{n_views}: batch {batch_idx}/{len(loader)}", flush=True)

    return torch.cat(all_logits, dim=0)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = str(cfg.training.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Pass the exact best checkpoint, e.g. "
            "training.checkpoint_path=checkpoints/video_swin_t_best.pt"
        )

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")

    model = build_model_from_checkpoint(ckpt)
    load_state_dict_strict_or_clean(model, ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model loaded on {device}", flush=True)

    # Use checkpoint metadata, not the current CLI config, for inference compatibility.
    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.pretrained))
    transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest", None)
    if manifest_cfg:
        video_names = load_manifest_video_names(Path(str(manifest_cfg)).resolve())
        samples = collect_unlabeled_video_samples(test_root, video_names)
    else:
        samples = collect_unlabeled_video_samples(test_root, None)
        video_names = [p.name for p, _ in samples]

    batch_size = int(cfg.training.get("eval_batch_size", cfg.training.batch_size))
    # For TTA, avoid too many workers if your filesystem is unstable.
    num_workers = int(cfg.training.get("num_workers", 4))
    n_views = int(cfg.dataset.get("tta_temporal_views", 5))
    n_views = max(1, n_views)

    print(
        f"Starting inference: clips={len(samples)} | num_frames={num_frames} | "
        f"batch_size={batch_size} | TTA temporal views={n_views}",
        flush=True,
    )

    summed_logits: Optional[torch.Tensor] = None
    for view_idx in range(n_views):
        view_logits = logits_for_temporal_view(
            model=model,
            samples=samples,
            test_root=test_root,
            transform=transform,
            num_frames=num_frames,
            view_idx=view_idx,
            n_views=n_views,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        summed_logits = view_logits if summed_logits is None else summed_logits + view_logits

    if summed_logits is None:
        raise RuntimeError("No logits produced.")

    avg_logits = summed_logits / float(n_views)
    predictions = avg_logits.argmax(dim=1).tolist()

    if len(predictions) != len(video_names):
        raise RuntimeError(f"Prediction count {len(predictions)} != video count {len(video_names)}")

    val_accuracy = None
    if ckpt.get("val_accuracy") is not None:
        try:
            val_accuracy = float(ckpt["val_accuracy"])
        except Exception:
            val_accuracy = None

    output_path = make_submission_path(Path(cfg.dataset.submission_output).resolve(), val_accuracy)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing submission: {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            writer.writerow([name, int(pred)])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
