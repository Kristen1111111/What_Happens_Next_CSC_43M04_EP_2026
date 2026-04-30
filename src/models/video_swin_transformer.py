from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VideoSwinClassifier(nn.Module):
    """
    Wrapper Torchvision Video Swin.

    Accepts input:
      - (B, T, C, H, W) from your dataset
      - (B, C, T, H, W)

    Internally converts to (B, C, T, H, W), expected by torchvision video models.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        arch: str = "swin3d_s",
        weights: str = "KINETICS400_V1",
        dropout: float = 0.45,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
        reset_head: bool = True,
        head_hidden_mult: float = 1.0,
    ) -> None:
        super().__init__()

        from torchvision.models import video as tv_video

        arch = str(arch)
        weights = str(weights)

        if arch == "swin3d_t":
            model_fn = tv_video.swin3d_t
            weights_enum = tv_video.Swin3D_T_Weights
        elif arch == "swin3d_s":
            model_fn = tv_video.swin3d_s
            weights_enum = tv_video.Swin3D_S_Weights
        elif arch == "swin3d_b":
            model_fn = tv_video.swin3d_b
            weights_enum = tv_video.Swin3D_B_Weights
        else:
            raise ValueError(f"Unknown Video Swin arch: {arch}")

        weights_obj = None
        if pretrained:
            if weights.upper() in ("DEFAULT", "TRUE"):
                weights_obj = weights_enum.DEFAULT
            else:
                try:
                    weights_obj = getattr(weights_enum, weights)
                except AttributeError as exc:
                    available = [x.name for x in weights_enum]
                    raise ValueError(
                        f"Unknown weights {weights!r} for {arch}. Available: {available}"
                    ) from exc

        self.backbone = model_fn(weights=weights_obj)

        # Torchvision Video Swin uses .head as final Linear.
        old_head = self.backbone.head
        if not isinstance(old_head, nn.Linear):
            raise TypeError(f"Expected backbone.head to be nn.Linear, got {type(old_head)}")

        in_features = old_head.in_features

        if reset_head:
            hidden = max(in_features // 2, int(in_features * float(head_hidden_mult)))
            self.backbone.head = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, int(num_classes)),
            )
        else:
            self.backbone.head = nn.Linear(in_features, int(num_classes))

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False

        # Optional partial unfreeze. This is conservative and only affects the final
        # feature blocks if torchvision exposes them through backbone.features.
        if int(unfreeze_last_n_blocks) > 0 and hasattr(self.backbone, "features"):
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.head.parameters():
                param.requires_grad = True

            blocks = list(self.backbone.features.children())
            for block in blocks[-int(unfreeze_last_n_blocks):]:
                for param in block.parameters():
                    param.requires_grad = True

    @staticmethod
    def _to_b_c_t_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected 5D video tensor, got shape {tuple(video.shape)}"
            )

        # Dataset gives (B, T, C, H, W). Convert to (B, C, T, H, W).
        if video.shape[2] in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()

        # Already (B, C, T, H, W).
        if video.shape[1] in (1, 3):
            return video

        raise ValueError(
            f"Cannot infer video format from shape {tuple(video.shape)}. "
            "Expected (B,T,C,H,W) or (B,C,T,H,W)."
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = self._to_b_c_t_h_w(video)
        return self.backbone(video)


def build_video_swin_transformer_classifier(
    num_classes: int,
    pretrained: bool = True,
    arch: str = "swin3d_s",
    weights: str = "KINETICS400_V1",
    dropout: float = 0.45,
    freeze_backbone: bool = False,
    unfreeze_last_n_blocks: int = 0,
    reset_head: bool = True,
    head_hidden_mult: float = 1.0,
) -> VideoSwinClassifier:
    return VideoSwinClassifier(
        num_classes=int(num_classes),
        pretrained=bool(pretrained),
        arch=str(arch),
        weights=str(weights),
        dropout=float(dropout),
        freeze_backbone=bool(freeze_backbone),
        unfreeze_last_n_blocks=int(unfreeze_last_n_blocks),
        reset_head=bool(reset_head),
        head_hidden_mult=float(head_hidden_mult),
    )
