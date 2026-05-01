#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("pretrain_ibot.py requires timm: pip install timm") from exc

from dataset.frame_ssl_dataset import FrameSSLDataset, collect_frames_from_roots
from utils import set_seed


class MultiCropTransform:
    """Two global views + optional local views for DINO/iBOT-style SSL."""

    def __init__(self, image_size: int = 224, local_size: int = 96, local_crops: int = 0) -> None:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.global_transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            normalize,
        ])
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            T.RandomSolarize(threshold=128, p=0.2),
            T.ToTensor(),
            normalize,
        ])
        self.local_crops = int(local_crops)
        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.08, 0.35), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transform(image), self.global_transform_2(image)]
        for _ in range(self.local_crops):
            crops.append(self.local_transform(image))
        return crops


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 8192, hidden_dim: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class MaskedTimmViT(nn.Module):
    """Timm ViT with optional masked patch tokens, returning CLS and patch tokens."""

    def __init__(self, vit_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.embed_dim = int(self.vit.num_features)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mostly follows timm VisionTransformer.forward_features.
        v = self.vit
        x = v.patch_embed(x)
        b, n, d = x.shape
        if mask is not None:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != (b, n):
                raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with patch tokens {(b, n)}")
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), x)
        cls = v.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = v.pos_drop(x + v.pos_embed[:, : n + 1])
        x = v.patch_drop(x) if hasattr(v, "patch_drop") else x
        x = v.norm_pre(x) if hasattr(v, "norm_pre") else x
        for blk in v.blocks:
            x = blk(x)
        x = v.norm(x)
        return x[:, 0], x[:, 1:]


def random_patch_mask(batch: int, num_patches: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    if mask_ratio <= 0:
        return torch.zeros(batch, num_patches, dtype=torch.bool, device=device)
    n_mask = max(1, int(num_patches * mask_ratio))
    noise = torch.rand(batch, num_patches, device=device)
    idx = noise.argsort(dim=1)[:, :n_mask]
    mask = torch.zeros(batch, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


class IBOTStudentTeacher(nn.Module):
    def __init__(self, vit_name: str, pretrained: bool, out_dim: int, patch_out_dim: int) -> None:
        super().__init__()
        self.encoder = MaskedTimmViT(vit_name, pretrained=pretrained)
        dim = self.encoder.embed_dim
        self.cls_head = DINOHead(dim, out_dim=out_dim)
        self.patch_head = DINOHead(dim, out_dim=patch_out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        cls, patches = self.encoder(x, mask)
        return self.cls_head(cls), self.patch_head(patches), cls, patches


def dino_ce(student_logits: torch.Tensor, teacher_probs: torch.Tensor, student_temp: float) -> torch.Tensor:
    return torch.sum(-teacher_probs * F.log_softmax(student_logits / student_temp, dim=-1), dim=-1).mean()


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def cosine_value(base: float, final: float, step: int, total: int) -> float:
    if total <= 1:
        return final
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * step / total))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Support your Hydra setup where experiment configs sometimes appear under cfg.experiment.
    if "experiment" in cfg and cfg.experiment is not None:
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, cfg.experiment)
        OmegaConf.set_struct(cfg, False)

    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device = torch.device("cuda" if str(cfg.training.device) == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    ssl_roots = list(cfg.ibot.get("ssl_roots", []))
    if not ssl_roots:
        ssl_roots = [cfg.dataset.train_dir, cfg.dataset.val_dir, cfg.dataset.test_dir]
    frames = collect_frames_from_roots(ssl_roots)
    max_frames = cfg.ibot.get("max_frames", None)
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    print(f"SSL frames: {len(frames)}")

    transform = MultiCropTransform(
        image_size=int(cfg.ibot.get("image_size", 224)),
        local_size=int(cfg.ibot.get("local_size", 96)),
        local_crops=int(cfg.ibot.get("local_crops", 0)),
    )
    dataset = FrameSSLDataset(frames, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.ibot.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.training.num_workers) > 0,
        drop_last=True,
    )

    student = IBOTStudentTeacher(
        vit_name=str(cfg.ibot.vit_name),
        pretrained=bool(cfg.ibot.get("pretrained", True)),
        out_dim=int(cfg.ibot.get("out_dim", 8192)),
        patch_out_dim=int(cfg.ibot.get("patch_out_dim", 8192)),
    ).to(device)

    # Build teacher without deepcopy: weight_norm heads can break deepcopy in PyTorch.
    teacher = IBOTStudentTeacher(
        vit_name=str(cfg.ibot.vit_name),
        pretrained=bool(cfg.ibot.get("pretrained", True)),
        out_dim=int(cfg.ibot.get("out_dim", 8192)),
        patch_out_dim=int(cfg.ibot.get("patch_out_dim", 8192)),
    ).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()
    for p_teacher in teacher.parameters():
        p_teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.ibot.lr), weight_decay=float(cfg.ibot.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(cfg.training.get("amp", True))))

    epochs = int(cfg.ibot.epochs)
    total_steps = max(1, epochs * len(loader))
    warmup_steps = max(1, int(cfg.ibot.get("warmup_epochs", 1)) * len(loader))
    center_cls = torch.zeros(1, int(cfg.ibot.get("out_dim", 8192)), device=device)
    center_patch = torch.zeros(1, int(cfg.ibot.get("patch_out_dim", 8192)), device=device)
    center_momentum = float(cfg.ibot.get("center_momentum", 0.9))

    out_path = Path(str(cfg.ibot.get("checkpoint_path", "checkpoints/ibot_pretrain_best.pt"))).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.with_name(out_path.stem + "_last" + out_path.suffix)

    global_step = 0
    for epoch in range(epochs):
        student.train()
        running = 0.0
        progress = tqdm(loader, desc=f"iBOT {epoch + 1}/{epochs}", leave=False)
        for crops in progress:
            # crops is a list of tensors, length 2 + local_crops.
            crops = [c.to(device, non_blocking=True) for c in crops]
            global_crops = crops[:2]

            # LR warmup + cosine.
            if global_step < warmup_steps:
                lr = float(cfg.ibot.lr) * float(global_step + 1) / warmup_steps
            else:
                lr = cosine_value(float(cfg.ibot.lr), float(cfg.ibot.get("min_lr", 1e-6)), global_step - warmup_steps, max(1, total_steps - warmup_steps))
            wd = cosine_value(float(cfg.ibot.weight_decay), float(cfg.ibot.get("weight_decay_end", cfg.ibot.weight_decay)), global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = wd

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_cls_logits = []
                t_patch_logits = []
                for x in global_crops:
                    cls_logit, patch_logit, _, _ = teacher(x, mask=None)
                    t_cls_logits.append(cls_logit)
                    t_patch_logits.append(patch_logit)
                teacher_temp = float(cfg.ibot.get("teacher_temp", 0.04))
                t_cls_probs = [F.softmax((z - center_cls) / teacher_temp, dim=-1).detach() for z in t_cls_logits]
                t_patch_probs = [F.softmax((z - center_patch.unsqueeze(1)) / teacher_temp, dim=-1).detach() for z in t_patch_logits]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None and device.type == "cuda")):
                student_cls_logits = []
                student_patch_logits = []
                masks = []
                for x in crops:
                    # Determine patch count from global/local resolution.
                    with torch.no_grad():
                        n_patches = student.encoder.vit.patch_embed(x).shape[1]
                    m = random_patch_mask(x.size(0), n_patches, float(cfg.ibot.get("mask_ratio", 0.35)), device)
                    cls_logit, patch_logit, _, _ = student(x, mask=m)
                    student_cls_logits.append(cls_logit)
                    student_patch_logits.append(patch_logit)
                    masks.append(m)

                loss_cls = 0.0
                n_cls_terms = 0
                for s_idx, s_logits in enumerate(student_cls_logits):
                    for t_idx, t_probs in enumerate(t_cls_probs):
                        if s_idx == t_idx:
                            continue
                        loss_cls = loss_cls + dino_ce(s_logits, t_probs, float(cfg.ibot.get("student_temp", 0.1)))
                        n_cls_terms += 1
                loss_cls = loss_cls / max(1, n_cls_terms)

                # Patch loss only on the two global crops where teacher patch grids match.
                loss_patch = 0.0
                n_patch_terms = 0
                for i in range(2):
                    s_patch = student_patch_logits[i]
                    t_patch = t_patch_probs[1 - i]
                    m = masks[i]
                    if m.any():
                        loss_patch = loss_patch + dino_ce(s_patch[m], t_patch[m], float(cfg.ibot.get("student_temp", 0.1)))
                        n_patch_terms += 1
                loss_patch = loss_patch / max(1, n_patch_terms)
                loss = loss_cls + float(cfg.ibot.get("patch_loss_weight", 1.0)) * loss_patch

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.ibot.get("clip_grad", 3.0)))
            scaler.step(optimizer)
            scaler.update()

            mom = cosine_value(float(cfg.ibot.get("ema_momentum", 0.996)), float(cfg.ibot.get("ema_momentum_end", 1.0)), global_step, total_steps)
            update_ema(student, teacher, mom)

            with torch.no_grad():
                batch_center = torch.cat(t_cls_logits, dim=0).mean(dim=0, keepdim=True)
                center_cls.mul_(center_momentum).add_(batch_center, alpha=1.0 - center_momentum)
                patch_center = torch.cat([z.reshape(-1, z.shape[-1]) for z in t_patch_logits], dim=0).mean(dim=0, keepdim=True)
                center_patch.mul_(center_momentum).add_(patch_center, alpha=1.0 - center_momentum)

            running += float(loss.item())
            global_step += 1
            progress.set_postfix(loss=f"{running / max(1, global_step % len(loader)):.4f}", lr=f"{lr:.2e}", mom=f"{mom:.4f}")

        payload = {
            "student_encoder": student.encoder.vit.state_dict(),
            "teacher_encoder": teacher.encoder.vit.state_dict(),
            "epoch": epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "vit_name": str(cfg.ibot.vit_name),
        }
        torch.save(payload, last_path)
        torch.save(payload, out_path)
        print(f"Epoch {epoch + 1}/{epochs} | saved iBOT checkpoint to {out_path}")


if __name__ == "__main__":
    main()
