from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install timm first: pip install timm") from exc


class DropPath(nn.Module):
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


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=min(0.15, dropout), batch_first=True)
        self.dp1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dp1(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(64, dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x).softmax(dim=1)
        return (x * w).sum(dim=1)


@dataclass
class IBOTVideoConfig:
    num_classes: int
    num_frames: int = 16
    vit_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    ibot_checkpoint: str | None = None
    freeze_encoder: bool = False
    unfreeze_last_n_blocks: int = 0
    embed_dim: int = 768
    temporal_depth: int = 4
    temporal_heads: int = 12
    temporal_mlp_ratio: float = 4.0
    num_query_tokens: int = 4
    dropout: float = 0.30
    drop_path_rate: float = 0.10
    head_hidden_mult: float = 1.5
    temporal_mask_prob: float = 0.05


class IBOTViTVideoClassifier(nn.Module):
    """Frame-level iBOT/ViT encoder + temporal Transformer classifier.

    Expected input shape from your dataset:
      - (B, T, C, H, W), or
      - (B, C, T, H, W)
    """

    def __init__(self, cfg: IBOTVideoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = int(cfg.num_frames)
        self.temporal_mask_prob = float(cfg.temporal_mask_prob)

        self.encoder = timm.create_model(
            cfg.vit_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
        )
        encoder_dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        self.encoder_dim = encoder_dim

        if cfg.ibot_checkpoint:
            self._load_ibot_checkpoint(cfg.ibot_checkpoint)

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Optional final block unfreeze for cheap domain adaptation.
            n = int(cfg.unfreeze_last_n_blocks)
            if n > 0 and hasattr(self.encoder, "blocks"):
                for block in self.encoder.blocks[-n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for name in ("norm", "fc_norm"):
                    m = getattr(self.encoder, name, None)
                    if m is not None:
                        for p in m.parameters():
                            p.requires_grad = True

        self.projector = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.num_frames, cfg.embed_dim))
        self.query_tokens = nn.Parameter(torch.zeros(1, int(cfg.num_query_tokens), cfg.embed_dim))

        dpr = torch.linspace(0, cfg.drop_path_rate, int(cfg.temporal_depth)).tolist()
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(cfg.embed_dim, cfg.temporal_heads, cfg.temporal_mlp_ratio, cfg.dropout, float(dpr[i]))
            for i in range(int(cfg.temporal_depth))
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.pool = AttentionPool(cfg.embed_dim)
        hidden = max(cfg.embed_dim, int(cfg.embed_dim * cfg.head_hidden_mult))
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, int(cfg.num_classes)),
        )
        self._init_new_layers()

    def _init_new_layers(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for m in [self.projector, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _load_ibot_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"iBOT checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state: Dict[str, Any]
        if isinstance(ckpt, dict):
            for key in ("student_encoder", "teacher_encoder", "encoder", "model_state_dict", "state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            else:
                state = ckpt
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

        cleaned = {}
        for k, v in state.items():
            kk = k
            for prefix in ("module.", "student.", "teacher.", "encoder.", "backbone."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            cleaned[kk] = v
        msg = self.encoder.load_state_dict(cleaned, strict=False)
        print(f"Loaded iBOT checkpoint from {ckpt_path}")
        print(f"  missing keys: {len(msg.missing_keys)} | unexpected keys: {len(msg.unexpected_keys)}")

    @staticmethod
    def _to_b_t_c_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got {tuple(video.shape)}")
        if video.shape[1] in (1, 3) and video.shape[2] not in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def _match_num_frames(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        if t == self.num_frames:
            return video
        if t > self.num_frames:
            idx = torch.linspace(0, t - 1, self.num_frames, device=video.device).long()
            return video.index_select(1, idx)
        pad = video[:, -1:].expand(b, self.num_frames - t, c, h, w)
        return torch.cat([video, pad], dim=1)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_features(frames)
        if isinstance(feats, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooled"):
                if key in feats:
                    return feats[key]
            if "x_norm_patchtokens" in feats:
                return feats["x_norm_patchtokens"].mean(dim=1)
        if feats.ndim == 3:
            return feats[:, 0]
        if feats.ndim == 2:
            return feats
        return feats.flatten(2).mean(dim=-1)

    def _temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.temporal_mask_prob <= 0:
            return x
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype) > self.temporal_mask_prob)
        all_masked = keep.sum(dim=1, keepdim=True) == 0
        keep = torch.where(all_masked, torch.ones_like(keep), keep)
        return x * keep / keep.float().mean(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = self._match_num_frames(self._to_b_t_c_h_w(video))
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frame_feats = self._encode_frames(frames).reshape(b, t, -1)
        x = self.projector(frame_feats) + self.temporal_pos[:, :t]
        x = self._temporal_mask(x)
        q = self.query_tokens.expand(b, -1, -1)
        x = torch.cat([q, x], dim=1)
        for block in self.temporal_blocks:
            x = block(x)
        x = self.norm(x)
        return self.classifier(self.pool(x))


def build_ibot_vit_video_classifier(
    num_classes: int,
    num_frames: int,
    pretrained: bool = True,
    vit_name: str = "vit_base_patch16_224",
    ibot_checkpoint: str | None = None,
    freeze_encoder: bool = False,
    unfreeze_last_n_blocks: int = 0,
    embed_dim: int = 768,
    temporal_depth: int = 4,
    temporal_heads: int = 12,
    temporal_mlp_ratio: float = 4.0,
    num_query_tokens: int = 4,
    dropout: float = 0.30,
    drop_path_rate: float = 0.10,
    head_hidden_mult: float = 1.5,
    temporal_mask_prob: float = 0.05,
) -> IBOTViTVideoClassifier:
    cfg = IBOTVideoConfig(
        num_classes=int(num_classes),
        num_frames=int(num_frames),
        vit_name=str(vit_name),
        pretrained=bool(pretrained),
        ibot_checkpoint=ibot_checkpoint if ibot_checkpoint not in (None, "", "null", "None") else None,
        freeze_encoder=bool(freeze_encoder),
        unfreeze_last_n_blocks=int(unfreeze_last_n_blocks),
        embed_dim=int(embed_dim),
        temporal_depth=int(temporal_depth),
        temporal_heads=int(temporal_heads),
        temporal_mlp_ratio=float(temporal_mlp_ratio),
        num_query_tokens=int(num_query_tokens),
        dropout=float(dropout),
        drop_path_rate=float(drop_path_rate),
        head_hidden_mult=float(head_hidden_mult),
        temporal_mask_prob=float(temporal_mask_prob),
    )
    return IBOTViTVideoClassifier(cfg)
