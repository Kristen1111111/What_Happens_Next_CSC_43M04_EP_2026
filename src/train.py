"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from utils import build_transforms, set_seed, split_train_val
from models.cnn_transformer import CNNTransformer 
from models.video_transformer import VideoTransformer
from models.video_transformer import build_video_transformer
from models.vl_jepa_video import build_vl_jepa_video_classifier
from dataset.video_augmentation import (
    VideoAugmentation,
    VideoAugmentationTransform,
    LabelSmoothingCrossEntropy,
    mixup_criterion,
)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "cnn_transformer":
        return CNNTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            hidden_dim=int(cfg.model.get("hidden_dim", 512)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.2)),
        ) 
    if name == "video_transformer":
        return build_video_transformer(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained,
            depth=int(cfg.model.get("depth", 12)),
            embed_dim=int(cfg.model.get("embed_dim", 768)),
            num_heads=int(cfg.model.get("num_heads", 12)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "vl_jepa_video":
        return build_vl_jepa_video_classifier(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained,
            x_encoder_name=str(cfg.model.get("x_encoder_name", "vit_small_patch16_224")),
            freeze_x_encoder=bool(cfg.model.get("freeze_x_encoder", False)),
            predictor_dim=int(cfg.model.get("predictor_dim", 512)),
            target_dim=int(cfg.model.get("target_dim", 512)),
            predictor_depth=int(cfg.model.get("predictor_depth", 4)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            num_query_tokens=int(cfg.model.get("num_query_tokens", 4)),
            dropout=float(cfg.model.get("dropout", 0.2)),
            logit_scale_init=float(cfg.model.get("logit_scale_init", 10.0)),
        )
    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    augmenter: VideoAugmentation | None = None,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Train {epoch + 1}/{total_epochs}",
        leave=False,
    )

    for video_batch, labels in progress_bar:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Augmentations sur tensor (GPU) si activées
        if augmenter is not None:
            result = augmenter(video_batch, labels)
            if isinstance(result, tuple):
                # MixUp activé
                video_batch, labels_a, labels_b, lam = result
                logits = model(video_batch)
                loss = mixup_criterion(loss_fn, logits, labels_a, labels_b, lam)
                predictions = logits.argmax(dim=1)
                correct += int((predictions == labels_a).sum().item())
            else:
                video_batch = result
                logits = model(video_batch)
                loss = loss_fn(logits, labels)
                predictions = logits.argmax(dim=1)
                correct += int((predictions == labels).sum().item())
        else:
            logits = model(video_batch)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)

        average_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        progress_bar.set_postfix(loss=f"{average_loss:.4f}", acc=f"{accuracy:.4f}")

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Val   {epoch + 1}/{total_epochs}",
        leave=False,
    )

    for video_batch, labels in progress_bar:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        average_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        progress_bar.set_postfix(loss=f"{average_loss:.4f}", acc=f"{accuracy:.4f}")

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)

    # Loss : label smoothing si cfg.training.label_smoothing > 0, sinon CrossEntropy standard
    smoothing = float(cfg.training.get("label_smoothing", 0.0))
    if smoothing > 0.0:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    # Scheduler cosine optionnel (activé si cfg.training.use_scheduler: true)
    scheduler = None
    if cfg.training.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.training.epochs),
            eta_min=1e-6,
        )

    # Augmentations sur tensor GPU (activées si cfg.training.use_augmentation: true)
    augmenter = None
    if cfg.training.get("use_augmentation", False):
        augmenter = VideoAugmentation(
            p_temporal_jitter=float(cfg.training.get("aug_temporal_jitter", 0.3)),
            p_frame_drop=float(cfg.training.get("aug_frame_drop", 0.2)),
            p_erasing=float(cfg.training.get("aug_erasing", 0.5)),
            p_mixup=float(cfg.training.get("aug_mixup", 0.3)),
            mixup_alpha=float(cfg.training.get("aug_mixup_alpha", 0.2)),
        )

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch, int(cfg.training.epochs), augmenter,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, epoch, int(cfg.training.epochs)
        )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()