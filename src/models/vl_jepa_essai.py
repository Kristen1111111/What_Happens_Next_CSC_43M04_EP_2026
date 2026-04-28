"""Strong VL-JEPA-style video classifier.

Drop-in replacement for ``models/vl_jepa_video.py`` used by your original
``train.py``.  It exposes the same factory function:

    build_vl_jepa_video_classifier(...)

Expected input shape from the dataset is either:
    - (B, T, C, H, W)
    - (B, C, T, H, W)

The module uses a pretrained timm image encoder frame-by-frame, then a stronger
video head: temporal positional embeddings, learnable query tokens, a deep
Transformer encoder, attention pooling, and a larger MLP classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "models/vl_jepa_video.py requires timm. Install it with: pip install timm"
    ) from exc


class DropPath(nn.Module):
    """Stochastic depth, kept local to avoid extra dependencies."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with stochastic depth."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class AttentionPool(nn.Module):
    """Learns which temporal/query tokens matter for classification."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.score(x).softmax(dim=1)
        return (x * weights).sum(dim=1)


@dataclass
class VLJEPAConfig:
    num_classes: int
    num_frames: int = 8
    pretrained: bool = True
    x_encoder_name: str = "vit_base_patch16_224"
    freeze_x_encoder: bool = False
    predictor_dim: int = 1024
    target_dim: int = 1024
    predictor_depth: int = 6
    num_heads: int = 16
    num_query_tokens: int = 8
    dropout: float = 0.30
    logit_scale_init: float = 10.0
    drop_path_rate: float = 0.10
    temporal_mask_prob: float = 0.10
    head_hidden_mult: float = 1.0


class StrongVLJEPAVideoClassifier(nn.Module):
    """Pretrained frame encoder + strong temporal JEPA-style classifier head."""

    def __init__(self, cfg: VLJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = int(cfg.num_frames)
        self.temporal_mask_prob = float(cfg.temporal_mask_prob)

        # num_classes=0 makes timm return features instead of classifier logits.
        self.x_encoder = timm.create_model(
            cfg.x_encoder_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        encoder_dim = int(self.x_encoder.num_features)

        if cfg.freeze_x_encoder:
            for p in self.x_encoder.parameters():
                p.requires_grad = False

        # Project image features into the stronger video latent space.
        self.frame_projector = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, cfg.predictor_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.predictor_dim, cfg.target_dim),
        )

        self.temporal_pos = nn.Parameter(torch.zeros(1, self.num_frames, cfg.target_dim))
        self.query_tokens = nn.Parameter(
            torch.zeros(1, int(cfg.num_query_tokens), cfg.target_dim)
        )

        dpr = torch.linspace(0, cfg.drop_path_rate, int(cfg.predictor_depth)).tolist()
        self.temporal_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.target_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=4.0,
                    dropout=cfg.dropout,
                    attn_dropout=min(0.15, cfg.dropout),
                    drop_path=float(dpr[i]),
                )
                for i in range(int(cfg.predictor_depth))
            ]
        )
        self.norm = nn.LayerNorm(cfg.target_dim)
        self.pool = AttentionPool(cfg.target_dim)

        # Extra robust classification head.
        head_hidden = max(cfg.target_dim, int(cfg.target_dim * cfg.head_hidden_mult))
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.target_dim),
            nn.Linear(cfg.target_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(head_hidden, cfg.target_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.target_dim // 2, cfg.num_classes),
        )

        # Kept for compatibility with your previous config; useful if you later add
        # contrastive/prototype losses, harmless for CE-only training.
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.logit_scale_init)).log())

        self._init_video_head()

    def _init_video_head(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for m in [self.frame_projector, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @staticmethod
    def _to_b_t_c_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"Expected a 5D video tensor, got shape {tuple(video.shape)}. "
                "Expected (B,T,C,H,W) or (B,C,T,H,W)."
            )
        # If second dimension is RGB channels, convert (B,C,T,H,W) -> (B,T,C,H,W).
        if video.shape[1] in (1, 3) and video.shape[2] not in (1, 3):
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def _match_num_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Robustly handles videos with T different from cfg.num_frames."""
        b, t, c, h, w = video.shape
        if t == self.num_frames:
            return video
        if t > self.num_frames:
            # Uniform temporal subsampling.
            idx = torch.linspace(0, t - 1, self.num_frames, device=video.device).long()
            return video.index_select(dim=1, index=idx)
        # Pad by repeating the last frame.
        pad = video[:, -1:].expand(b, self.num_frames - t, c, h, w)
        return torch.cat([video, pad], dim=1)

    def _temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Light frame-feature dropout: encourages robustness to missing/noisy frames."""
        if not self.training or self.temporal_mask_prob <= 0.0:
            return x
        b, t, d = x.shape
        keep = torch.rand(b, t, 1, device=x.device, dtype=x.dtype) > self.temporal_mask_prob
        # Avoid masking all frames in one sample.
        all_masked = keep.sum(dim=1, keepdim=True) == 0
        keep = torch.where(all_masked, torch.ones_like(keep), keep)
        return x * keep / keep.float().mean(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = self._to_b_t_c_h_w(video)
        video = self._match_num_frames(video)

        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)

        # Encode frames independently with the pretrained image backbone.
        frame_features = self.x_encoder(frames)
        if frame_features.ndim > 2:
            frame_features = frame_features.flatten(2).mean(dim=-1)
        frame_features = frame_features.reshape(b, t, -1)

        x = self.frame_projector(frame_features)
        x = x + self.temporal_pos[:, :t]
        x = self._temporal_mask(x)

        queries = self.query_tokens.expand(b, -1, -1)
        x = torch.cat([queries, x], dim=1)

        for block in self.temporal_blocks:
            x = block(x)

        x = self.norm(x)
        # Pool over query tokens and frame tokens. AttentionPool will learn the best mix.
        pooled = self.pool(x)
        logits = self.classifier(pooled)
        return logits


def build_vl_jepa_video_classifier(
    num_classes: int,
    num_frames: int,
    pretrained: bool = True,
    x_encoder_name: str = "vit_base_patch16_224",
    freeze_x_encoder: bool = False,
    predictor_dim: int = 1024,
    target_dim: int = 1024,
    predictor_depth: int = 6,
    num_heads: int = 16,
    num_query_tokens: int = 8,
    dropout: float = 0.30,
    logit_scale_init: float = 10.0,
    drop_path_rate: float = 0.10,
    temporal_mask_prob: float = 0.10,
    head_hidden_mult: float = 1.0,
) -> StrongVLJEPAVideoClassifier:
    """Factory used by your original train.py."""
    cfg = VLJEPAConfig(
        num_classes=int(num_classes),
        num_frames=int(num_frames),
        pretrained=bool(pretrained),
        x_encoder_name=str(x_encoder_name),
        freeze_x_encoder=bool(freeze_x_encoder),
        predictor_dim=int(predictor_dim),
        target_dim=int(target_dim),
        predictor_depth=int(predictor_depth),
        num_heads=int(num_heads),
        num_query_tokens=int(num_query_tokens),
        dropout=float(dropout),
        logit_scale_init=float(logit_scale_init),
        drop_path_rate=float(drop_path_rate),
        temporal_mask_prob=float(temporal_mask_prob),
        head_hidden_mult=float(head_hidden_mult),
    )
    return StrongVLJEPAVideoClassifier(cfg)
