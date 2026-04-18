"""ATTENTION FULL IA C'EST POUR TEST"""


from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        attn = self.score(x).squeeze(-1)          # (B, T)
        attn = torch.softmax(attn, dim=1)         # (B, T)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # (B, D)
        return pooled


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Backbone image
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection vers dimension temporelle
        self.proj = nn.Linear(feature_dim, hidden_dim)

        # Embedding de position temporelle
        self.max_frames = 128
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_frames, hidden_dim))

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
        self.pool = AttentionPooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        if t > self.max_frames:
            raise ValueError(
                f"Number of frames {t} exceeds max_frames={self.max_frames}"
            )

        # Backbone image par frame
        x = x.view(b * t, c, h, w)         # (B*T, C, H, W)
        feats = self.backbone(x)           # (B*T, feature_dim)
        feats = self.proj(feats)           # (B*T, hidden_dim)
        feats = feats.view(b, t, -1)       # (B, T, hidden_dim)

        # Ajout position temporelle
        feats = feats + self.pos_embedding[:, :t, :]

        # Modélisation temporelle
        feats = self.transformer(feats)    # (B, T, hidden_dim)
        feats = self.norm(feats)

        # Pooling attention
        pooled = self.pool(feats)          # (B, hidden_dim)

        logits = self.classifier(pooled)   # (B, num_classes)
        return logits