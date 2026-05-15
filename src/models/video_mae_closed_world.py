import torch
import torch.nn as nn
import torch.nn.functional as F


class TubeletEmbed(nn.Module):
    """
    Patch embedding spatio-temporel.
    Input:  B,T,C,H,W ou B,C,T,H,W
    Output: B,N,D
    """
    def __init__(self, in_chans=3, embed_dim=384, tubelet_size=2, patch_size=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        x = self.proj(x)                 # B,D,T',H',W'
        x = x.flatten(2).transpose(1, 2) # B,N,D
        return x


class VideoMAEClosedWorld(nn.Module):
    """
    Modèle vidéo from scratch, closed-world strict.
    Aucun poids pré-entraîné externe.
    """
    def __init__(
        self,
        num_classes=33,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.20,
        tubelet_size=2,
        patch_size=16,
        num_frames=16,
        image_size=224,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        self.patch_embed = TubeletEmbed(
            in_chans=3,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
        )

        t = num_frames // tubelet_size
        h = image_size // patch_size
        w = image_size // patch_size
        self.num_tokens = t * h * w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor with 5 dims, got {x.shape}")

        # Dataset probable: B,T,C,H,W. Conv3D attend B,C,T,H,W.
        if x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.patch_embed(x)  # B,N,D
        b, n, _ = x.shape

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # Sécurité si num_frames diffère légèrement.
            pos_tokens = self.pos_embed[:, 1:].transpose(1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=n, mode="linear", align_corners=False)
            pos_tokens = pos_tokens.transpose(1, 2)
            pos = torch.cat([self.pos_embed[:, :1], pos_tokens], dim=1)
            x = x + pos

        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.head(features)
        return logits
