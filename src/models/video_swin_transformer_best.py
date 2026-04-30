"""Competitive Video Swin classifier for folder-of-frames video datasets.

Put this file in: src/models/video_swin_transformer.py

Input accepted:
    - (B, T, C, H, W) from your VideoFrameDataset
    - (B, C, T, H, W) already channel-first video

Backbone:
    torchvision Video Swin Transformer pretrained on Kinetics-400.

Recommended for Kaggle:
    arch=swin3d_s, weights=KINETICS400_V1, num_frames=32,
    MLP head + differential LR in train2_kaggle_best.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from torchvision.models.video import (
        swin3d_t,
        swin3d_s,
        swin3d_b,
        Swin3D_T_Weights,
        Swin3D_S_Weights,
        Swin3D_B_Weights,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Video Swin requires a recent torchvision with swin3d_t/s/b. "
        "Try: pip install -U torchvision"
    ) from exc


def _resolve_weights(arch: str, weights: str | None, pretrained: bool):
    if not pretrained:
        return None

    weights_name = "DEFAULT" if weights in (None, "", "default", "DEFAULT") else str(weights)
    arch = str(arch).lower()

    if arch == "swin3d_t":
        return Swin3D_T_Weights[weights_name]
    if arch == "swin3d_s":
        return Swin3D_S_Weights[weights_name]
    if arch == "swin3d_b":
        return Swin3D_B_Weights[weights_name]

    raise ValueError(f"Unknown Video Swin arch: {arch}. Use swin3d_t, swin3d_s or swin3d_b.")


def _make_backbone(arch: str, weights_obj: Any):
    arch = str(arch).lower()
    if arch == "swin3d_t":
        return swin3d_t(weights=weights_obj)
    if arch == "swin3d_s":
        return swin3d_s(weights=weights_obj)
    if arch == "swin3d_b":
        return swin3d_b(weights=weights_obj)
    raise ValueError(f"Unknown Video Swin arch: {arch}")


class VideoSwinClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        arch: str = "swin3d_s",
        weights: str = "KINETICS400_V1",
        dropout: float = 0.45,
        head_hidden_mult: float = 1.0,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
        reset_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.arch = str(arch)

        weights_obj = _resolve_weights(arch, weights, pretrained)
        self.model = _make_backbone(arch, weights_obj)

        if not hasattr(self.model, "head") or not isinstance(self.model.head, nn.Linear):
            raise RuntimeError("Unexpected torchvision Video Swin structure: missing Linear head.")

        in_features = int(self.model.head.in_features)
        hidden = max(in_features, int(in_features * float(head_hidden_mult)))

        if reset_head:
            self.model.head = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.num_classes),
            )
        else:
            # Mostly for debugging. Kinetics head has 400 classes, so reset_head should
            # stay true for this competition.
            self.model.head = nn.Linear(in_features, self.num_classes)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad_(False)
            for param in self.model.head.parameters():
                param.requires_grad_(True)

            # Optional partial unfreeze of the deepest feature blocks.
            if int(unfreeze_last_n_blocks) > 0 and hasattr(self.model, "features"):
                for block in list(self.model.features)[-int(unfreeze_last_n_blocks):]:
                    for param in block.parameters():
                        param.requires_grad_(True)

    @staticmethod
    def _to_b_c_t_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected 5D video tensor, got {tuple(video.shape)}. "
                "Expected (B,T,C,H,W) or (B,C,T,H,W)."
            )
        # Dataset returns (B,T,C,H,W). Torchvision Video Swin expects (B,C,T,H,W).
        if video.shape[2] in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video.contiguous()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = self._to_b_c_t_h_w(video)
        return self.model(video)


def build_video_swin_transformer_classifier(
    num_classes: int,
    pretrained: bool = True,
    arch: str = "swin3d_s",
    weights: str = "KINETICS400_V1",
    dropout: float = 0.45,
    head_hidden_mult: float = 1.0,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
    reset_head: bool = True,
) -> VideoSwinClassifier:
    return VideoSwinClassifier(
        num_classes=int(num_classes),
        pretrained=bool(pretrained),
        arch=str(arch),
        weights=str(weights),
        dropout=float(dropout),
        head_hidden_mult=float(head_hidden_mult),
        freeze_backbone=bool(freeze_backbone),
        unfreeze_last_n_blocks=int(unfreeze_last_n_blocks),
        reset_head=bool(reset_head),
    )
