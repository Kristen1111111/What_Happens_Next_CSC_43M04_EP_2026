#!/usr/bin/env python3
"""Kaggle submission with temporal TTA and optional checkpoint ensembling.

Put this file in: src/create_submission.py

Single checkpoint:
    python create_submission.py experiment=video_swin_competitive_essai \
      training.checkpoint_path=checkpoints/video_swin_s_best.pt \
      dataset.tta_temporal_views=7

Ensemble, comma-separated:
    python create_submission.py experiment=video_swin_competitive_essai \
      inference.checkpoints=checkpoints/video_swin_s_best.pt,checkpoints/video_swin_t_seed123_best.pt \
      dataset.tta_temporal_views=7
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset
from train2_kaggle_best import build_model
from utils import build_transforms, set_seed


def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader]


def _index_video_folders(test_root: Path) -> Dict[str, Path]:
    test_root = test_root.resolve()
    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(f"Duplicate video folder name {name!r}: {index[name]} and {p}")
            index[name] = p
            dirs.remove(name)
    return index


def resolve_video_dirs(test_root: Path, video_names: List[str]) -> List[Path]:
    index = _index_video_folders(test_root)
    missing = [name for name in video_names if name not in index]
    if missing:
        sample = ", ".join(repr(m) for m in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(f"{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}")
    return [index[name] for name in video_names]


def discover_all_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    index = _index_video_folders(test_root)
    video_names = sorted(index.keys())
    return video_names, [index[name] for name in video_names]


def make_submission_path(base_output_path: Path, val_accuracy: Optional[float], n_models: int, n_views: int) -> Path:
    if base_output_path.is_dir():
        base_output_path = base_output_path / "submission.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_suffix = f"acc{val_accuracy:.4f}" if val_accuracy is not None else "accNA"
    filename = f"{base_output_path.stem}_{acc_suffix}_m{n_models}_tta{n_views}_{timestamp}{base_output_path.suffix or '.csv'}"
    return base_output_path.parent / filename


def parse_checkpoints(cfg: DictConfig) -> List[Path]:
    raw = cfg.get("inference", {}).get("checkpoints", None)
    if raw is None or str(raw).strip() == "":
        raw = str(cfg.training.checkpoint_path)

    paths = [Path(x.strip()).resolve() for x in str(raw).split(",") if x.strip()]
    if not paths:
        raise ValueError("No checkpoints provided.")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return paths


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError("Checkpoint has no saved Hydra config. Train with the current train2.py.")
    saved_cfg = OmegaConf.create(ckpt["config"])
    model = build_model(saved_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def logits_for_model_and_view(
    model: torch.nn.Module,
    video_dirs: List[Path],
    test_root: Path,
    num_frames: int,
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    view_idx: int,
    n_views: int,
) -> torch.Tensor:
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=transform,
        sample_list=sample_list,
        temporal_view_index=view_idx,
        temporal_num_views=n_views,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    chunks: List[torch.Tensor] = []
    for batch_idx, (video_batch, _labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device, non_blocking=True)
        logits = model(video_batch).float().cpu()
        chunks.append(logits)
        if batch_idx == len(loader) or batch_idx % max(1, len(loader) // 5) == 0:
            print(f"    view {view_idx + 1}/{n_views}: batch {batch_idx}/{len(loader)}", flush=True)
    return torch.cat(chunks, dim=0)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = str(cfg.training.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_paths = parse_checkpoints(cfg)
    print("Using checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    checkpoints: List[Dict[str, Any]] = [torch.load(p, map_location="cpu") for p in checkpoint_paths]
    models = [build_model_from_checkpoint(ckpt, device) for ckpt in checkpoints]

    # Use first checkpoint as source of truth for preprocessing.
    first_ckpt = checkpoints[0]
    num_frames = int(first_ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(first_ckpt.get("pretrained", cfg.model.pretrained))
    transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        video_names = load_manifest_video_names(Path(str(manifest_cfg)).resolve())
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Resolved {len(video_dirs)} test videos.")

    n_views = int(cfg.dataset.get("tta_temporal_views", 7))
    batch_size = int(cfg.training.get("eval_batch_size", cfg.training.batch_size))
    num_workers = int(cfg.training.num_workers)

    final_logits: Optional[torch.Tensor] = None
    for model_idx, model in enumerate(models, start=1):
        print(f"Running model {model_idx}/{len(models)}")
        model_logits: Optional[torch.Tensor] = None
        for view_idx in range(n_views):
            view_logits = logits_for_model_and_view(
                model=model,
                video_dirs=video_dirs,
                test_root=test_root,
                num_frames=num_frames,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                view_idx=view_idx,
                n_views=n_views,
            )
            model_logits = view_logits if model_logits is None else model_logits + view_logits
        model_logits = model_logits / float(n_views)
        final_logits = model_logits if final_logits is None else final_logits + model_logits

    final_logits = final_logits / float(len(models))
    predictions = final_logits.argmax(dim=1).tolist()

    if len(predictions) != len(video_names):
        raise RuntimeError(f"Prediction count {len(predictions)} != number of videos {len(video_names)}")

    best_val = None
    vals = []
    for ckpt in checkpoints:
        try:
            vals.append(float(ckpt.get("val_accuracy")))
        except Exception:
            pass
    if vals:
        best_val = max(vals)

    output_path = make_submission_path(
        Path(cfg.dataset.submission_output).resolve(),
        val_accuracy=best_val,
        n_models=len(models),
        n_views=n_views,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            writer.writerow([name, int(pred)])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
