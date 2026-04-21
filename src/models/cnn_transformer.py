from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MaskedMeanPooling(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, D)
        if mask is None:
            return x.mean(dim=1)

        mask = mask.unsqueeze(-1).float()          # (B, T, 1)
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)    # (B, 1)
        return x.sum(dim=1) / denom


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.35,
        max_frames: int = 128,
        freeze_backbone: bool = True,
        unfreeze_layer4: bool = True,
    ) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

            if unfreeze_layer4:
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = True

        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.max_frames = max_frames
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_frames, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = MaskedMeanPooling()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        if t == 0:
            raise ValueError("Input sequence has zero frames")
        if t > self.max_frames:
            raise ValueError(f"Number of frames {t} exceeds max_frames={self.max_frames}")

        x = x.reshape(b * t, c, h, w)
        feats = self.backbone(x)              # (B*T, feature_dim)
        feats = self.proj(feats)              # (B*T, hidden_dim)
        feats = feats.reshape(b, t, -1)       # (B, T, hidden_dim)

        feats = feats + self.pos_embedding[:, :t, :]
        feats = self.input_dropout(feats)

        src_key_padding_mask = None
        if mask is not None:
            # mask: True = frame valide
            src_key_padding_mask = ~mask      # True = à masquer pour le transformer

        feats = self.transformer(
            feats,
            src_key_padding_mask=src_key_padding_mask,
        )
        feats = self.norm(feats)

        pooled = self.pool(feats, mask=mask)
        logits = self.classifier(pooled)
        return logits