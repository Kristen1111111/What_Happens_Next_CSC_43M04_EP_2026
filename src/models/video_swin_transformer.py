"""Video Swin Transformer classifier for folder-of-frames video datasets.

Drop-in model file for ``src/models/video_swin_transformer.py``.

Expected input from your ``VideoFrameDataset``:
    - (B, T, C, H, W), which is what your dataset returns after batching
    - also accepts (B, C, T, H, W)

The wrapped torchvision Video Swin models expect (B, C, T, H, W), so this file
handles the permutation internally.

Recommended dependency:
    pip install -U torchvision

Factory used from train.py/train2.py:
    build_video_swin_transformer_classifier(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models.video import swin3d_b, swin3d_s, swin3d_t
    from torchvision.models.video import Swin3D_B_Weights, Swin3D_S_Weights, Swin3D_T_Weights
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "video_swin_transformer.py requires torchvision with video Swin models. "
        "Try: pip install -U torchvision"
    ) from exc


@dataclass
class VideoSwinConfig:
    num_classes: int
    arch: str = "swin3d_t"  # swin3d_t, swin3d_s, swin3d_b
    pretrained: bool = True
    weights: str = "KINETICS400_V1"
    dropout: float = 0.35
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 0
    reset_head: bool = True


def _resolve_weights(arch: str, pretrained: bool, weights_name: str):
    """Return the matching torchvision weights enum or None.

    This is deliberately defensive because torchvision names differ slightly
    between versions. If the requested enum is unavailable, it falls back to the
    default pretrained weights for the selected architecture.
    """
    if not pretrained:
        return None

    arch = arch.lower()
    weights_name = str(weights_name or "DEFAULT")

    if arch == "swin3d_t":
        enum = Swin3D_T_Weights
    elif arch == "swin3d_s":
        enum = Swin3D_S_Weights
    elif arch == "swin3d_b":
        enum = Swin3D_B_Weights
    else:
        raise ValueError(f"Unknown Video Swin arch: {arch!r}")

    if weights_name.upper() in {"DEFAULT", "TRUE", "KINETICS400_V1"}:
        # DEFAULT is safest across torchvision versions.
        try:
            return enum.DEFAULT
        except Exception:
            pass

    try:
        return getattr(enum, weights_name)
    except Exception:
        print(
            f"Warning: weights {weights_name!r} not found for {arch}; using DEFAULT weights.",
            flush=True,
        )
        return enum.DEFAULT


def _make_torchvision_swin(arch: str, weights):
    arch = arch.lower()
    if arch == "swin3d_t":
        return swin3d_t(weights=weights)
    if arch == "swin3d_s":
        return swin3d_s(weights=weights)
    if arch == "swin3d_b":
        return swin3d_b(weights=weights)
    raise ValueError(f"Unknown Video Swin arch: {arch!r}. Use swin3d_t, swin3d_s or swin3d_b.")


def _freeze_all_but_last_blocks(model: nn.Module, unfreeze_last_n_blocks: int) -> None:
    """Freeze most of the backbone while optionally unfreezing the last blocks.

    This is useful if your dataset is small. For maximum score with enough GPU,
    prefer freeze_backbone=False.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Always train the classification head.
    for p in model.head.parameters():
        p.requires_grad = True

    if unfreeze_last_n_blocks <= 0:
        return

    # Torchvision Video Swin stores main stages in model.features. This loop is
    # intentionally generic so it does not break across minor torchvision changes.
    candidates = []
    if hasattr(model, "features"):
        for child in model.features.children():
            candidates.append(child)

    for module in candidates[-int(unfreeze_last_n_blocks):]:
        for p in module.parameters():
            p.requires_grad = True


class VideoSwinTransformerClassifier(nn.Module):
    """Torchvision Video Swin wrapper with your dataset's tensor layout."""

    def __init__(self, cfg: VideoSwinConfig) -> None:
        super().__init__()
        self.cfg = cfg
        weights = _resolve_weights(cfg.arch, cfg.pretrained, cfg.weights)
        self.backbone = _make_torchvision_swin(cfg.arch, weights)

        if not hasattr(self.backbone, "head") or not isinstance(self.backbone.head, nn.Linear):
            raise RuntimeError("Unexpected torchvision Video Swin structure: missing Linear head.")

        in_features = int(self.backbone.head.in_features)
        if cfg.reset_head:
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=float(cfg.dropout)),
                nn.Linear(in_features, int(cfg.num_classes)),
            )
        else:
            # Still force correct class count.
            self.backbone.head = nn.Linear(in_features, int(cfg.num_classes))

        if cfg.freeze_backbone:
            _freeze_all_but_last_blocks(self.backbone, int(cfg.unfreeze_last_n_blocks))

    @staticmethod
    def _to_b_c_t_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected 5D video tensor, got {tuple(video.shape)}. "
                "Expected (B,T,C,H,W) or (B,C,T,H,W)."
            )

        # Your dataset returns (B,T,C,H,W). Torchvision video models want (B,C,T,H,W).
        if video.shape[2] in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()

        # Already (B,C,T,H,W).
        if video.shape[1] in (1, 3):
            return video.contiguous()

        raise ValueError(
            f"Could not infer channel dimension from video shape {tuple(video.shape)}. "
            "Expected RGB channel dimension of size 3."
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = self._to_b_c_t_h_w(video)
        return self.backbone(video)


def build_video_swin_transformer_classifier(
    num_classes: int,
    pretrained: bool = True,
    arch: str = "swin3d_t",
    weights: str = "KINETICS400_V1",
    dropout: float = 0.35,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
    reset_head: bool = True,
) -> VideoSwinTransformerClassifier:
    cfg = VideoSwinConfig(
        num_classes=int(num_classes),
        arch=str(arch),
        pretrained=bool(pretrained),
        weights=str(weights),
        dropout=float(dropout),
        freeze_backbone=bool(freeze_backbone),
        unfreeze_last_n_blocks=int(unfreeze_last_n_blocks),
        reset_head=bool(reset_head),
    )
    return VideoSwinTransformerClassifier(cfg)
