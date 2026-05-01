from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_frame_files(root: Path) -> Iterable[Path]:
    root = root.resolve()
    if not root.is_dir():
        return
    for dirpath, _dirs, files in os.walk(root):
        base = Path(dirpath)
        for name in files:
            p = base / name
            if p.suffix.lower() in IMG_EXTS:
                yield p


def collect_frames_from_roots(roots: Sequence[str | Path]) -> List[Path]:
    frames: List[Path] = []
    seen = set()
    for r in roots:
        root = Path(r).expanduser().resolve()
        if not root.is_dir():
            print(f"Warning: SSL root does not exist, skipped: {root}")
            continue
        for p in _iter_frame_files(root):
            if p not in seen:
                frames.append(p)
                seen.add(p)
    frames.sort(key=lambda x: str(x))
    if not frames:
        raise RuntimeError(f"No frames found under SSL roots: {roots}")
    return frames


class FrameSSLDataset(Dataset):
    def __init__(self, frames: Sequence[str | Path], transform: Callable) -> None:
        self.frames = [Path(p) for p in frames]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        path = self.frames[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.transform(im)
