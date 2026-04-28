"""
VideoFrameDataset with CSV-label support and deterministic temporal views.

Key fixes for Kaggle-style workflows:
- training/validation labels can come from CSV files (video_name,class_idx),
  instead of relying only on class-folder prefixes;
- video folders are indexed once by their `video_*` folder name;
- inference can use multiple deterministic temporal views for test-time augmentation.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


Sample = Tuple[Path, int]


def _list_frame_paths(video_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)


def _parse_class_index(class_dir_name: str) -> Optional[int]:
    match = re.match(r"^(\d+)_", class_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def _index_video_folders(root_dir: Path) -> Dict[str, Path]:
    """Map video folder name -> full path, without descending into frame folders."""
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(root_dir, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise RuntimeError(f"Duplicate video folder {name!r}: {index[name]} and {p}")
            index[name] = p
            # Frames live inside video dirs; no need to walk all images.
            dirs.remove(name)
    return index


def collect_video_samples(root_dir: Path) -> List[Sample]:
    """
    Folder-based fallback. Expects:
        root_dir/000_ClassName/video_xxx/frame.jpg
    Label is parsed from leading numeric prefix, or from sorted class-folder order.
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: List[Sample] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            if len(_list_frame_paths(video_dir)) == 0:
                continue
            samples.append((video_dir.resolve(), int(class_index)))

    if len(samples) == 0:
        raise RuntimeError(f"No video folders with frames under {root_dir}")
    return samples


def collect_video_samples_from_csv(root_dir: Path, csv_path: Path) -> List[Sample]:
    """
    CSV-based source of truth. CSV must contain:
        video_name,class_idx
    This avoids class-index mismatches caused by folder ordering or renamed folders.
    """
    root_dir = root_dir.resolve()
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")

    index = _index_video_folders(root_dir)
    samples: List[Sample] = []
    missing: List[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        required = {"video_name", "class_idx"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{csv_path} must contain columns {sorted(required)}; got {reader.fieldnames}")

        for row in reader:
            video_name = row["video_name"].strip()
            if not video_name:
                continue
            video_dir = index.get(video_name)
            if video_dir is None:
                missing.append(video_name)
                continue
            samples.append((video_dir, int(row["class_idx"])))

    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"{len(missing)} CSV video(s) not found under {root_dir}: {preview}{extra}"
        )
    if len(samples) == 0:
        raise RuntimeError(f"No usable samples loaded from {csv_path}")
    return samples


def collect_unlabeled_video_samples(root_dir: Path, video_names: Optional[List[str]] = None) -> List[Sample]:
    """For test inference. Returns dummy label 0 while preserving requested order."""
    index = _index_video_folders(root_dir)
    if video_names is None:
        video_names = sorted(index.keys())
    samples: List[Sample] = []
    missing: List[str] = []
    for name in video_names:
        p = index.get(name)
        if p is None:
            missing.append(name)
        else:
            samples.append((p, 0))
    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise FileNotFoundError(f"Missing test videos under {root_dir}: {preview}{extra}")
    return samples


def labels_from_samples(samples: Iterable[Sample]) -> List[int]:
    return [int(label) for _path, label in samples]


def infer_num_classes(samples: Iterable[Sample], configured_num_classes: Optional[int] = None) -> int:
    labels = labels_from_samples(samples)
    if len(labels) == 0:
        if configured_num_classes is None:
            raise ValueError("Cannot infer num_classes from an empty sample list.")
        return int(configured_num_classes)
    required = max(labels) + 1
    if configured_num_classes is None:
        return required
    configured = int(configured_num_classes)
    if configured < required:
        raise ValueError(
            f"model.num_classes={configured} is too small for labels up to {max(labels)}. "
            f"Use at least {required}."
        )
    return configured


def _pick_frame_indices(
    num_available: int,
    num_frames: int,
    temporal_view_index: int = 0,
    temporal_num_views: int = 1,
) -> List[int]:
    """
    Deterministic frame indices.
    - view 0/1 behaves like the original uniform sampling;
    - with multiple temporal views, the sampling window is shifted across time.
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_available == 1:
        return [0] * num_frames

    temporal_num_views = max(1, int(temporal_num_views))
    temporal_view_index = max(0, min(int(temporal_view_index), temporal_num_views - 1))

    if temporal_num_views == 1 or num_available <= num_frames:
        positions = torch.linspace(0, num_available - 1, steps=num_frames)
        return [int(round(float(x))) for x in positions]

    # Multi-view inference: use a shifted temporal window, then sample uniformly inside it.
    window = max(num_frames, int(round(num_available * 0.80)))
    window = min(window, num_available)
    max_start = num_available - window
    start = int(round(max_start * temporal_view_index / max(1, temporal_num_views - 1)))
    positions = torch.linspace(start, start + window - 1, steps=num_frames)
    return [int(round(float(x))) for x in positions]


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable[[Image.Image], torch.Tensor],
        sample_list: Optional[List[Sample]] = None,
        temporal_view_index: int = 0,
        temporal_num_views: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = int(num_frames)
        self.transform = transform
        self.temporal_view_index = int(temporal_view_index)
        self.temporal_num_views = int(temporal_num_views)

        if sample_list is None:
            self.samples = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        indices = _pick_frame_indices(
            len(frame_paths),
            self.num_frames,
            temporal_view_index=self.temporal_view_index,
            temporal_num_views=self.temporal_num_views,
        )

        frames: List[torch.Tensor] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
            frames.append(self.transform(rgb_image))

        video_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return video_tensor, label_tensor
