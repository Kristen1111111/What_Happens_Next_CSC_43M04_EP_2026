"""
VideoTransformer: TimeSformer-style architecture with divided space-time attention,
pretrained ViT-B/16 backbone (ImageNet-21k) and a temporal anticipation head.

Architecture overview:
    Input: (B, T, C, H, W)
    ↓
    Patch Embedding (16×16 patches per frame) → (B, T, N_patches, D)
    + Temporal positional encoding + Spatial positional encoding
    ↓
    N Transformer blocks, each with:
        - Temporal Self-Attention  (attend across time for same spatial patch)
        - Spatial Self-Attention   (attend across patches within a frame)
        - MLP feed-forward (D → 4D → D)
        - LayerNorm + residual connections
    ↓
    CLS token → Temporal Anticipation Head:
        LayerNorm → Linear(D→512) → GELU → Dropout → Linear(512→num_classes)

Why it wins for anticipation (partial video → predict future action):
    1. Divided space-time attention captures *when* things move (temporal),
       and *what* is in the frame (spatial), separately — efficient and expressive.
    2. Pretrained ViT-B/16 (timm, ImageNet-21k) gives rich visual priors.
    3. Causal-style temporal pooling (later frames weighted more) helps
       capture the trajectory of motion right before the cut.
    4. Label smoothing + dropout regularize the small SSv2 subset.

Drop-in replacement: same forward(video_batch: (B,T,C,H,W)) → logits: (B, num_classes)
interface as CNNBaseline and CNNLSTM.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # pip install timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split each frame into non-overlapping 16×16 patches and project to D dims.
    Reuses a pretrained 2D patch embedding when available.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        x = self.proj(x)          # (B*T, D, H/P, W/P)
        x = x.flatten(2)          # (B*T, D, N)
        x = x.transpose(1, 2)     # (B*T, N, D)
        return x


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention across the time dimension for each spatial position.
    Each of the N_patches tokens independently attends across T timesteps.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D) — N spatial patches, T frames
        B, T, N, D = x.shape

        # Reshape: treat each patch position independently over time
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)   # (B*N, T, D)

        qkv = self.qkv(x).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B*N, heads, T, head_dim)
        q, k, v = qkv.unbind(0)                           # each: (B*N, heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B*N, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)  # (B*N, T, D)
        out = self.proj(out)

        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)       # (B, T, N, D)
        return out


class SpatialAttention(nn.Module):
    """
    Standard multi-head self-attention across spatial patches within each frame.
    Includes the extra CLS token dimension if present.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # Each frame independently
        x_flat = x.reshape(B * T, N, D)                          # (B*T, N, D)

        qkv = self.qkv(x_flat).reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                   # (B*T, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.proj(out)

        return out.reshape(B, T, N, D)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DividedSpaceTimeBlock(nn.Module):
    """
    One transformer block with divided space-time attention (TimeSformer design).
    Order: temporal attn → spatial attn → MLP, all with pre-norm and residual.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, attn_dropout)

        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, attn_dropout)

        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

        self.drop_path = nn.Identity()  # could replace with DropPath for deeper nets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        # 1. Temporal attention
        x = x + self.temporal_attn(self.norm_temporal(x))
        # 2. Spatial attention
        x = x + self.spatial_attn(self.norm_spatial(x))
        # 3. MLP
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class VideoTransformer(nn.Module):
    """
    Divided space-time Video Transformer for action anticipation.

    Args:
        num_classes:   Number of action classes (50 for the challenge).
        num_frames:    T — number of input frames per video clip.
        img_size:      Spatial resolution of each frame (224).
        patch_size:    ViT patch size (16 → 14×14 = 196 patches per frame).
        embed_dim:     Token dimensionality (768 for ViT-Base).
        depth:         Number of transformer blocks (12 for ViT-Base).
        num_heads:     Attention heads (12 for ViT-Base).
        mlp_ratio:     MLP hidden-dim expansion factor (4.0).
        dropout:       MLP / output dropout.
        attn_dropout:  Attention weight dropout.
        pretrained:    Load ViT-B/16 ImageNet-21k spatial weights from timm.
        head_hidden:   Hidden dim of the classification head (512).
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pretrained: bool = True,
        head_hidden: int = 512,
    ) -> None:
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2   # 196 for 224/16

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # --- CLS token (used for global representation) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional encodings ---
        # Spatial: 1D over patches + CLS
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        # Temporal: 1D over T frames (no CLS dimension)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # --- Temporal anticipation head ---
        # Causal weighting: later frames contribute more (captures motion trend)
        weights = torch.linspace(0.5, 1.0, num_frames)    # e.g. 8 frames → [0.5…1.0]
        self.register_buffer("temporal_weights", weights)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # --- Weight initialisation ---
        self._init_weights()

        # --- Load pretrained ViT-B/16 spatial weights (timm) ---
        if pretrained:
            self._load_pretrained_vit()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _load_pretrained_vit(self):
        """
        Transfer spatial weights from a pretrained ViT (timm).
        Le modèle est choisi automatiquement selon embed_dim :
            384 → vit_small_patch16_224  (~22M params, tient sur 11GB)
            768 → vit_base_patch16_224   (~86M params, nécessite >16GB)
        Temporal attention est initialisée depuis spatial attention (TimeSformer).
        """
        if not _TIMM_AVAILABLE:
            print(
                "[VideoTransformer] timm not found. Run `pip install timm` for pretrained "
                "weights. Training from scratch instead."
            )
            return

        # Sélection automatique selon embed_dim
        timm_model_name = {
            384: "vit_small_patch16_224",
            768: "vit_base_patch16_224",
        }.get(self.embed_dim)

        if timm_model_name is None:
            print(
                f"[VideoTransformer] Pas de modèle pretrained pour embed_dim={self.embed_dim} "
                f"(supportés: 384=ViT-Small, 768=ViT-Base). Training from scratch."
            )
            return

        try:
            vit = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0,   # remove head
            )
        except Exception as exc:
            print(f"[VideoTransformer] Could not load pretrained ViT: {exc}. Skipping.")
            return

        print(f"[VideoTransformer] Chargement de {timm_model_name} (embed_dim={self.embed_dim})")

        # Patch embedding
        self.patch_embed.proj.weight.data.copy_(vit.patch_embed.proj.weight.data)
        self.patch_embed.proj.bias.data.copy_(vit.patch_embed.proj.bias.data)

        # Spatial positional embedding: vit has (1, N+1, D) with CLS at 0
        if vit.pos_embed.shape == self.spatial_pos_embed.shape:
            self.spatial_pos_embed.data.copy_(vit.pos_embed.data)

        # CLS token
        self.cls_token.data.copy_(vit.cls_token.data)

        # Transformer blocks: spatial attn qkv/proj + MLP
        for i, (our_block, vit_block) in enumerate(zip(self.blocks, vit.blocks)):
            # Spatial attention weights
            our_block.spatial_attn.qkv.weight.data.copy_(vit_block.attn.qkv.weight.data)
            our_block.spatial_attn.qkv.bias.data.copy_(vit_block.attn.qkv.bias.data)
            our_block.spatial_attn.proj.weight.data.copy_(vit_block.attn.proj.weight.data)
            our_block.spatial_attn.proj.bias.data.copy_(vit_block.attn.proj.bias.data)

            # Warm-start temporal attention from spatial weights
            our_block.temporal_attn.qkv.weight.data.copy_(vit_block.attn.qkv.weight.data)
            our_block.temporal_attn.qkv.bias.data.copy_(vit_block.attn.qkv.bias.data)
            our_block.temporal_attn.proj.weight.data.copy_(vit_block.attn.proj.weight.data)
            our_block.temporal_attn.proj.bias.data.copy_(vit_block.attn.proj.bias.data)

            # Layer norms
            our_block.norm_spatial.weight.data.copy_(vit_block.norm1.weight.data)
            our_block.norm_spatial.bias.data.copy_(vit_block.norm1.bias.data)
            our_block.norm_temporal.weight.data.copy_(vit_block.norm1.weight.data)
            our_block.norm_temporal.bias.data.copy_(vit_block.norm1.bias.data)

            # MLP
            our_block.mlp.fc1.weight.data.copy_(vit_block.mlp.fc1.weight.data)
            our_block.mlp.fc1.bias.data.copy_(vit_block.mlp.fc1.bias.data)
            our_block.mlp.fc2.weight.data.copy_(vit_block.mlp.fc2.weight.data)
            our_block.mlp.fc2.bias.data.copy_(vit_block.mlp.fc2.bias.data)
            our_block.norm_mlp.weight.data.copy_(vit_block.norm2.weight.data)
            our_block.norm_mlp.bias.data.copy_(vit_block.norm2.bias.data)

        # Final norm
        if hasattr(vit, "norm"):
            self.norm.weight.data.copy_(vit.norm.weight.data)
            self.norm.bias.data.copy_(vit.norm.bias.data)

        del vit
        print("[VideoTransformer] Loaded pretrained ViT-B/16 spatial weights from timm.")

    # ------------------------------------------------------------------
    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_batch: (B, T, C, H, W) — T frames per clip
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape

        # ---- 1. Patch embedding (process all frames at once) ----
        frames = video_batch.reshape(B * T, C, H, W)
        patches = self.patch_embed(frames)                 # (B*T, N, D)
        N = patches.shape[1]                               # 196

        # ---- 2. Add spatial positional embedding (without CLS for now) ----
        # spatial_pos_embed is (1, N+1, D); index [1:] skips CLS position
        patches = patches + self.spatial_pos_embed[:, 1:, :]

        # ---- 3. Add temporal positional embedding ----
        # Broadcast temporal PE over spatial patches
        temporal_pe = self.temporal_pos_embed.unsqueeze(2)        # (1, T, 1, D)
        patches = patches.reshape(B, T, N, -1) + temporal_pe      # (B, T, N, D)

        patches = self.pos_drop(patches)

        # ---- 4. Transformer blocks ----
        x = patches   # (B, T, N, D)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)    # (B, T, N, D)

        # ---- 5. Temporal anticipation pooling ----
        # Causal weighting: weight each frame's mean-pooled representation
        # later frames carry more signal about the imminent action
        w = self.temporal_weights.to(x.dtype)              # (T,)
        w = w / w.sum()

        # Average over spatial patches per frame, then weighted temporal sum
        frame_feats = x.mean(dim=2)                        # (B, T, D)
        pooled = (frame_feats * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, D)

        # ---- 6. Classification head ----
        logits = self.head(pooled)                         # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Convenience factory (mirrors the pattern in train.py)
# ---------------------------------------------------------------------------

def build_video_transformer(
    num_classes: int,
    num_frames: int = 8,
    pretrained: bool = True,
    depth: int = 12,
    embed_dim: int = 768,
    num_heads: int = 12,
    dropout: float = 0.1,
    head_hidden: int = 512,
) -> VideoTransformer:
    return VideoTransformer(
        num_classes=num_classes,
        num_frames=num_frames,
        img_size=224,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=dropout,
        attn_dropout=0.0,
        pretrained=pretrained,
        head_hidden=head_hidden,
    )