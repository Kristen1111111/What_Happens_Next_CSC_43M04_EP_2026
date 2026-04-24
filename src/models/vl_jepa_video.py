"""
VL-JEPA-inspired video classifier for action recognition / anticipation.

Drop-in model for the current training pipeline:
    forward(video_batch: Tensor[B, T, C, H, W]) -> logits Tensor[B, num_classes]

This is not the full Meta VL-JEPA training recipe, because the provided train.py
expects supervised classification logits and a CrossEntropy-like loss. The model
keeps the core idea that matters for this codebase:

    visual video X_V -> compact visual embedding S_V
    predictor        -> predicted target/action embedding S_hat_Y
    class prototypes -> candidate target embeddings S_Y
    logits           -> cosine similarity(S_hat_Y, S_Y)

So the classification head is an embedding-space matching head rather than a
plain Linear(D, num_classes) head. This follows the discriminative / retrieval
use case of VL-JEPA while remaining compatible with train.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # pip install timm
    _TIMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    timm = None
    _TIMM_AVAILABLE = False


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAggregator(nn.Module):
    """Small causal-style temporal attention over frame embeddings."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = MLP(dim, hidden_dim=4 * dim, dropout=dropout)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        # frame_tokens: (B, T, D)
        x = self.norm(frame_tokens)
        # Causal mask: each time step sees only current/past frames.
        T = x.shape[1]
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = frame_tokens + attn_out
        x = x + self.mlp(x)
        return x


class VLJEPAVideoClassifier(nn.Module):
    """
    VL-JEPA-inspired classifier.

    Args:
        num_classes: number of labels.
        num_frames: expected number of frames per clip.
        x_encoder_name: timm ViT backbone. Recommended: vit_small_patch16_224.
        pretrained: load ImageNet pretrained weights when timm can provide them.
        freeze_x_encoder: freeze visual encoder for small datasets / low VRAM.
        predictor_dim: latent dimension used by the JEPA-style predictor.
        target_dim: shared target embedding dimension for predicted embedding and class prototypes.
        predictor_depth: number of TransformerEncoder layers in the predictor.
        num_heads: attention heads in temporal aggregator and predictor.
        num_query_tokens: learned target-query tokens. More tokens = richer predicted target.
        dropout: dropout in predictor and projection blocks.
        logit_scale_init: initial cosine-logit temperature. 10.0 is CLIP-like.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 8,
        x_encoder_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_x_encoder: bool = False,
        predictor_dim: int = 512,
        target_dim: int = 512,
        predictor_depth: int = 4,
        num_heads: int = 8,
        num_query_tokens: int = 4,
        dropout: float = 0.2,
        logit_scale_init: float = 10.0,
        img_size: int = 224,
        **_: object,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("VLJEPAVideoClassifier requires timm. Install with: pip install timm")

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.predictor_dim = predictor_dim
        self.target_dim = target_dim
        self.num_query_tokens = num_query_tokens

        self.x_encoder = timm.create_model(
            x_encoder_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.x_dim = int(getattr(self.x_encoder, "num_features", predictor_dim))

        if freeze_x_encoder:
            for p in self.x_encoder.parameters():
                p.requires_grad = False

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.x_dim),
            nn.Linear(self.x_dim, predictor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, predictor_dim))
        self.temporal_aggregator = TemporalAggregator(
            dim=predictor_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Target-query tokens play the role of a lightweight query-conditioned predictor.
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, predictor_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=4 * predictor_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=predictor_depth,
            norm=nn.LayerNorm(predictor_dim),
        )

        self.target_projection = nn.Sequential(
            nn.LayerNorm(predictor_dim),
            nn.Linear(predictor_dim, target_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(target_dim, target_dim),
        )

        # Learnable class embeddings are a supervised replacement for the PDF's Y-Encoder
        # candidate-answer embeddings, compatible with CrossEntropyLoss.
        self.class_prototypes = nn.Parameter(torch.empty(num_classes, target_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)).log())

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.class_prototypes, std=0.02)
        for module in [self.visual_projection, self.target_projection]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Return one compact feature vector per frame: (B*T, D_x)."""
        feats = self.x_encoder.forward_features(frames)

        if isinstance(feats, dict):
            # timm backbones may return dicts depending on model family.
            for key in ("x", "features", "last_hidden_state"):
                if key in feats:
                    feats = feats[key]
                    break
            else:
                feats = next(v for v in feats.values() if torch.is_tensor(v))

        if feats.ndim == 4:
            # CNN-like feature map: global average pool.
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            # ViT tokens: use CLS token when present, otherwise mean-pool tokens.
            feats = feats[:, 0] if feats.shape[1] > 1 else feats.mean(dim=1)
        elif feats.ndim != 2:
            raise RuntimeError(f"Unexpected x_encoder output shape: {tuple(feats.shape)}")

        return feats

    def encode_video(self, video_batch: torch.Tensor) -> torch.Tensor:
        """X-Encoder + Predictor: video -> normalized predicted target embedding."""
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)

        frame_feats = self._extract_frame_features(frames).reshape(B, T, self.x_dim)
        x = self.visual_projection(frame_feats)

        if T <= self.temporal_pos_embed.shape[1]:
            x = x + self.temporal_pos_embed[:, :T]
        else:
            # Supports inference with more frames than the training config.
            pe = F.interpolate(
                self.temporal_pos_embed.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pe

        x = self.temporal_aggregator(x)

        # Later frames matter more for anticipation.
        weights = torch.linspace(0.5, 1.0, T, device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        pooled_context = (x * weights.view(1, T, 1)).sum(dim=1, keepdim=True)

        query = self.query_tokens.expand(B, -1, -1)
        predictor_input = torch.cat([query, pooled_context, x], dim=1)
        pred_tokens = self.predictor(predictor_input)[:, : self.num_query_tokens]
        pred = pred_tokens.mean(dim=1)
        pred = self.target_projection(pred)
        return F.normalize(pred, dim=-1)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        pred = self.encode_video(video_batch)
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * pred @ prototypes.t()


def build_vl_jepa_video_classifier(
    num_classes: int,
    num_frames: int = 8,
    pretrained: bool = True,
    x_encoder_name: str = "vit_small_patch16_224",
    freeze_x_encoder: bool = False,
    predictor_dim: int = 512,
    target_dim: int = 512,
    predictor_depth: int = 4,
    num_heads: int = 8,
    num_query_tokens: int = 4,
    dropout: float = 0.2,
    logit_scale_init: float = 10.0,
    img_size: int = 224,
    **kwargs: object,
) -> VLJEPAVideoClassifier:
    return VLJEPAVideoClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        x_encoder_name=x_encoder_name,
        pretrained=pretrained,
        freeze_x_encoder=freeze_x_encoder,
        predictor_dim=predictor_dim,
        target_dim=target_dim,
        predictor_depth=predictor_depth,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        dropout=dropout,
        logit_scale_init=logit_scale_init,
        img_size=img_size,
        **kwargs,
    )
