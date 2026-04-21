from __future__ import annotations
import math
import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        qkv = self.qkv(x).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.proj(out)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        qkv = self.qkv(x_flat).reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.proj(out)
        return out.reshape(B, T, N, D)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DividedSpaceTimeBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, attn_dropout)
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, attn_dropout)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.temporal_attn(self.norm_temporal(x))
        x = x + self.spatial_attn(self.norm_spatial(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class VideoTransformer(nn.Module):
    def __init__(self, num_classes, num_frames=8, img_size=224, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.0, pretrained=True, head_hidden=512):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        weights = torch.linspace(0.5, 1.0, num_frames)
        self.register_buffer("temporal_weights", weights)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

        self._init_weights()
        if pretrained:
            self._load_pretrained_vit()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained_vit(self):
        if not _TIMM_AVAILABLE:
            print("[VideoTransformer] timm not found. Training from scratch.")
            return
        try:
            vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        except Exception as e:
            print(f"[VideoTransformer] Could not load pretrained ViT: {e}")
            return
        self.patch_embed.proj.weight.data.copy_(vit.patch_embed.proj.weight.data)
        self.patch_embed.proj.bias.data.copy_(vit.patch_embed.proj.bias.data)
        if vit.pos_embed.shape == self.spatial_pos_embed.shape:
            self.spatial_pos_embed.data.copy_(vit.pos_embed.data)
        self.cls_token.data.copy_(vit.cls_token.data)
        for our_block, vit_block in zip(self.blocks, vit.blocks):
            for attn in [our_block.spatial_attn, our_block.temporal_attn]:
                attn.qkv.weight.data.copy_(vit_block.attn.qkv.weight.data)
                attn.qkv.bias.data.copy_(vit_block.attn.qkv.bias.data)
                attn.proj.weight.data.copy_(vit_block.attn.proj.weight.data)
                attn.proj.bias.data.copy_(vit_block.attn.proj.bias.data)
            for norm in [our_block.norm_spatial, our_block.norm_temporal]:
                norm.weight.data.copy_(vit_block.norm1.weight.data)
                norm.bias.data.copy_(vit_block.norm1.bias.data)
            our_block.mlp.fc1.weight.data.copy_(vit_block.mlp.fc1.weight.data)
            our_block.mlp.fc1.bias.data.copy_(vit_block.mlp.fc1.bias.data)
            our_block.mlp.fc2.weight.data.copy_(vit_block.mlp.fc2.weight.data)
            our_block.mlp.fc2.bias.data.copy_(vit_block.mlp.fc2.bias.data)
            our_block.norm_mlp.weight.data.copy_(vit_block.norm2.weight.data)
            our_block.norm_mlp.bias.data.copy_(vit_block.norm2.bias.data)
        if hasattr(vit, "norm"):
            self.norm.weight.data.copy_(vit.norm.weight.data)
            self.norm.bias.data.copy_(vit.norm.bias.data)
        del vit
        print("[VideoTransformer] Loaded pretrained ViT-B/16 weights.")

    def forward(self, video_batch):
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)
        patches = self.patch_embed(frames)
        N = patches.shape[1]
        patches = patches + self.spatial_pos_embed[:, 1:, :]
        temporal_pe = self.temporal_pos_embed.unsqueeze(2)
        patches = patches.reshape(B, T, N, -1) + temporal_pe
        patches = self.pos_drop(patches)
        x = patches
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        w = self.temporal_weights.to(x.dtype)
        w = w / w.sum()
        frame_feats = x.mean(dim=2)
        pooled = (frame_feats * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)


def build_video_transformer(num_classes, num_frames=8, pretrained=True,
                             depth=12, embed_dim=768, num_heads=12,
                             dropout=0.1, head_hidden=512):
    return VideoTransformer(
        num_classes=num_classes, num_frames=num_frames, img_size=224,
        patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4.0, dropout=dropout, attn_dropout=0.0,
        pretrained=pretrained, head_hidden=head_hidden,
    )
