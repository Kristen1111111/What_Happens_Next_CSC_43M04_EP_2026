import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.video_mae_closed_world import VideoMAEClosedWorld


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class UnlabeledVideoFramesDataset(Dataset):
    def __init__(self, roots, num_frames=16, image_size=224, max_videos=None):
        self.roots = [Path(r) for r in roots]
        self.num_frames = num_frames

        self.videos = []
        for root in self.roots:
            if not root.exists():
                print(f"Warning: root does not exist: {root}")
                continue
            for p in sorted(root.rglob("video_*")):
                if p.is_dir():
                    frames = sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])
                    if len(frames) > 0:
                        self.videos.append((p, frames))

        if max_videos is not None:
            self.videos = self.videos[:max_videos]

        print(f"Unlabeled videos: {len(self.videos)}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.videos)

    def _sample_frames(self, frames):
        n = len(frames)
        if n >= self.num_frames:
            idx = torch.linspace(0, n - 1, self.num_frames).long().tolist()
            if n > self.num_frames:
                jitter = max(1, n // self.num_frames)
                idx = [min(n - 1, max(0, i + random.randint(-jitter // 2, jitter // 2))) for i in idx]
            return [frames[i] for i in idx]
        else:
            idx = list(range(n))
            while len(idx) < self.num_frames:
                idx.append(idx[-1])
            return [frames[i] for i in idx]

    def __getitem__(self, i):
        _, frames = self.videos[i]
        chosen = self._sample_frames(frames)

        imgs = []
        for f in chosen:
            img = Image.open(f).convert("RGB")
            imgs.append(self.transform(img))

        video = torch.stack(imgs, dim=0)  # T,C,H,W
        return video


class VideoMAEPretrainer(nn.Module):
    def __init__(self, encoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder.embed_dim

        self.reconstruct = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

    def forward(self, x):
        # x: B,T,C,H,W
        if x.shape[2] == 3:
            x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x_3d = x

        tokens = self.encoder.patch_embed(x_3d)  # B,N,D
        b, n, d = tokens.shape

        mask = torch.rand(b, n, device=tokens.device) < self.mask_ratio

        visible = tokens.clone()
        visible[mask] = 0.0

        cls = self.encoder.cls_token.expand(b, -1, -1)
        z = torch.cat([cls, visible], dim=1)

        if z.shape[1] == self.encoder.pos_embed.shape[1]:
            z = z + self.encoder.pos_embed

        z = self.encoder.blocks(z)
        z = self.encoder.norm(z)
        pred = self.reconstruct(z[:, 1:])

        loss = F.mse_loss(pred[mask], tokens.detach()[mask])
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="/Data/KristenLeGoat/train")
    parser.add_argument("--val-dir", default="/Data/KristenLeGoat/val")
    parser.add_argument("--test-dir", default="/Data/KristenLeGoat/test")
    parser.add_argument("--output", default="/Data/kristen.boitier/checkpoints/video_mae_cw_ssl_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnlabeledVideoFramesDataset(
        roots=[args.train_dir, args.val_dir, args.test_dir],
        num_frames=args.num_frames,
        max_videos=args.max_videos,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )

    encoder = VideoMAEClosedWorld(
        num_classes=33,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.15,
        tubelet_size=2,
        patch_size=16,
        num_frames=args.num_frames,
        image_size=224,
    )

    model = VideoMAEPretrainer(encoder, mask_ratio=args.mask_ratio).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(loader, desc=f"SSL {epoch}/{args.epochs}", dynamic_ncols=True)

        for videos in pbar:
            videos = videos.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(videos)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=f"{total_loss / max(1, n_steps):.5f}")

        avg_loss = total_loss / max(1, n_steps)
        print(f"Epoch {epoch}/{args.epochs} | ssl loss {avg_loss:.6f}", flush=True)

        ckpt = {
            "epoch": epoch,
            "ssl_loss": avg_loss,
            "model": model.encoder.state_dict(),
            "config": vars(args),
        }

        torch.save(ckpt, output)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output.with_name(output.stem + "_best.pt")
            torch.save(ckpt, best_path)
            print(f"Saved best SSL encoder to {best_path} (loss={best_loss:.6f})", flush=True)


if __name__ == "__main__":
    main()
