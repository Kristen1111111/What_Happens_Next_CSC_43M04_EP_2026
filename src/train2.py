"""
Improved video classifier training script.

Drop-in replacement for train.py. It keeps the same Hydra config structure, but adds:
- AdamW + weight decay
- AMP mixed precision
- gradient accumulation
- gradient clipping
- warmup + cosine or one-cycle LR schedulers
- optional class-balanced sampling
- optional class-weighted loss
- optional EMA validation/checkpointing
- optional torch.compile
- optional DataParallel for multi-GPU
- robust checkpoint metadata and resume support

Run from src/:
    python train_improved.py experiment=vl_jepa_video_strong
"""

from __future__ import annotations

import copy
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset.video_dataset import (
    VideoFrameDataset,
    collect_video_samples,
    collect_video_samples_from_csv,
)
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.video_transformer import build_video_transformer
from models.vl_jepa_video import build_vl_jepa_video_classifier
from utils import build_transforms, set_seed, split_train_val
from dataset.video_augmentation import (
    VideoAugmentation,
    LabelSmoothingCrossEntropy,
    mixup_criterion,
)


# -----------------------------
# Model factory
# -----------------------------

def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = str(cfg.model.name)
    num_classes = int(cfg.model.num_classes)
    pretrained = bool(cfg.model.pretrained)

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)

    if name == "cnn_lstm":
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 512)),
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
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.10)),
            temporal_mask_prob=float(cfg.model.get("temporal_mask_prob", 0.10)),
            head_hidden_mult=float(cfg.model.get("head_hidden_mult", 1.0)),
        )

    raise ValueError(f"Unknown model.name: {name}")


# -----------------------------
# Small utilities
# -----------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_sample_label(sample: Any) -> int | None:
    """Best-effort label extraction for class balancing."""
    if isinstance(sample, Mapping):
        for key in ("label", "class_idx", "target", "y"):
            if key in sample:
                return int(sample[key])

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        try:
            return int(sample[1])
        except Exception:
            return None

    if hasattr(sample, "label"):
        return int(sample.label)

    return None


def labels_from_samples(samples: Sequence[Any]) -> list[int] | None:
    labels = [get_sample_label(s) for s in samples]
    if any(y is None for y in labels):
        return None
    return [int(y) for y in labels]


def build_balanced_sampler(samples: Sequence[Any]) -> WeightedRandomSampler | None:
    labels = labels_from_samples(samples)
    if labels is None or len(labels) == 0:
        print("Class-balanced sampler disabled: could not infer labels from sample_list.")
        return None

    counts = Counter(labels)
    weights = torch.DoubleTensor([1.0 / counts[y] for y in labels])
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_class_weights(
    samples: Sequence[Any],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    labels = labels_from_samples(samples)
    if labels is None or len(labels) == 0:
        print("Class-weighted loss disabled: could not infer labels from sample_list.")
        return None

    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        if counts.get(c, 0) > 0:
            weights[c] = len(labels) / (num_classes * counts[c])

    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.to(device)


def make_loss_fn(
    cfg: DictConfig,
    train_samples: Sequence[Any],
    device: torch.device,
) -> nn.Module:
    smoothing = float(cfg.training.get("label_smoothing", 0.0))
    use_class_weights = bool(cfg.training.get("class_weighted_loss", False))

    weights = None
    if use_class_weights:
        weights = build_class_weights(train_samples, int(cfg.model.num_classes), device)

    if weights is not None or smoothing > 0.0:
        try:
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
        except TypeError:
            if weights is not None:
                print(
                    "Warning: class weights require a recent PyTorch CrossEntropyLoss; "
                    "ignoring weights."
                )
            return (
                LabelSmoothingCrossEntropy(smoothing=smoothing)
                if smoothing > 0.0
                else nn.CrossEntropyLoss()
            )

    return nn.CrossEntropyLoss()


def make_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.training.get("optimizer", "adamw")).lower()
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 0.05))

    decay, no_decay = [], []

    for name, param in unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(
                float(cfg.training.get("beta1", 0.9)),
                float(cfg.training.get("beta2", 0.999)),
            ),
            eps=float(cfg.training.get("eps", 1e-8)),
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)

    if optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def make_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    scheduler_name = str(cfg.training.get("scheduler", "cosine_warmup")).lower()
    epochs = int(cfg.training.epochs)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 5))
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    min_lr = float(cfg.training.get("min_lr", 1e-6))

    if scheduler_name in ("none", "false", "off"):
        return None, "none"

    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.training.lr),
            total_steps=total_steps,
            pct_start=float(cfg.training.get("onecycle_pct_start", 0.1)),
            div_factor=float(cfg.training.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(cfg.training.get("onecycle_final_div_factor", 1e4)),
        )
        return scheduler, "step"

    if scheduler_name == "cosine_warmup":
        base_lrs = [group["lr"] for group in optimizer.param_groups]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = min_lr / max(base_lrs[0], 1e-12)
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"

    if scheduler_name == "cosine_epoch" or bool(cfg.training.get("use_scheduler", False)):
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
            ),
            "epoch",
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


class ModelEMA:
    """Exponential Moving Average of model weights for steadier validation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(unwrap_model(model)).eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = unwrap_model(model).state_dict()
        ema_state = self.ema.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()

            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema.to(device)
        return self


# -----------------------------
# Train / eval loops
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    augmenter: VideoAugmentation | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_interval: str = "none",
    grad_accum_steps: int = 1,
    max_grad_norm: float = 0.0,
    ema: ModelEMA | None = None,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    grad_accum_steps = max(1, int(grad_accum_steps))
    use_amp = scaler is not None and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        data_loader,
        desc=f"Train {epoch + 1}/{total_epochs}",
        leave=False,
    )

    for step, (video_batch, labels) in enumerate(progress_bar):
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            effective_labels_for_acc = labels

            if augmenter is not None:
                result = augmenter(video_batch, labels)

                if isinstance(result, tuple):
                    video_batch, labels_a, labels_b, lam = result
                    logits = model(video_batch)
                    loss = mixup_criterion(loss_fn, logits, labels_a, labels_b, lam)
                    effective_labels_for_acc = labels_a
                else:
                    video_batch = result
                    logits = model(video_batch)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(video_batch)
                loss = loss_fn(logits, labels)

            loss_for_backward = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(data_loader)

        if should_step:
            if use_amp:
                scaler.unscale_(optimizer)

            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and scheduler_interval == "step":
                scheduler.step()

            if ema is not None:
                ema.update(model)

        running_loss += float(loss.item()) * labels.size(0)

        predictions = logits.argmax(dim=1)
        correct += int((predictions == effective_labels_for_acc).sum().item())
        total += labels.size(0)

        average_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        lr = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            acc=f"{accuracy:.4f}",
            lr=f"{lr:.2e}",
        )

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
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
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)

        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{correct / max(total, 1):.4f}",
        )

    return running_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: DictConfig,
    val_acc: float,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEMA | None = None,
    save_training_state: bool = False,
    use_ema_weights: bool = False,
) -> None:
    """
    Save a lightweight checkpoint compatible with create_submission.py.

    By default:
    - does NOT save optimizer state
    - does NOT save scheduler state
    - does NOT save ema_state_dict separately

    This keeps checkpoint files much smaller and avoids hitting the 30GB quota.

    If use_ema_weights=True and ema is available, EMA weights are saved directly
    as model_state_dict. That means create_submission.py can load the checkpoint
    normally without knowing anything about EMA.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_ema_weights and ema is not None:
        state_dict = unwrap_model(ema.ema).state_dict()
    else:
        state_dict = unwrap_model(model).state_dict()

    payload: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "model_name": str(cfg.model.name),
        "num_classes": int(cfg.model.num_classes),
        "pretrained": bool(cfg.model.pretrained),
        "num_frames": int(cfg.dataset.num_frames),
        "val_accuracy": float(val_acc),
        "epoch": int(epoch),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    # Heavy resume state is optional.
    # Keep this False for normal training if your quota is small.
    if save_training_state:
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

    tmp_path = str(path) + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


# -----------------------------
# Main
# -----------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    if bool(cfg.training.get("cudnn_benchmark", True)) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if bool(cfg.training.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device_str = str(cfg.training.device)

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()

    # Prefer official CSV manifests when available. This is critical for Kaggle:
    # folder order / folder prefix must never silently define another label mapping.
    train_labels_csv = cfg.dataset.get("train_labels_csv", None) or cfg.dataset.get("train_csv", None)
    val_labels_csv = cfg.dataset.get("val_labels_csv", None) or cfg.dataset.get("val_csv", None)

    if train_labels_csv:
        train_samples = collect_video_samples_from_csv(train_dir, Path(str(train_labels_csv)))

        if val_labels_csv:
            val_dir = Path(cfg.dataset.get("val_dir", cfg.dataset.train_dir)).resolve()
            val_samples = collect_video_samples_from_csv(val_dir, Path(str(val_labels_csv)))
        else:
            train_samples, val_samples = split_train_val(
                train_samples,
                val_ratio=float(cfg.dataset.val_ratio),
                seed=int(cfg.dataset.seed),
            )

        print(f"Using CSV labels: train={train_labels_csv} val={val_labels_csv}")

    else:
        all_samples = collect_video_samples(train_dir)
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=float(cfg.dataset.val_ratio),
            seed=int(cfg.dataset.seed),
        )
        print("Using folder-derived labels. For Kaggle, prefer dataset.train_labels_csv.")

    max_samples = cfg.dataset.get("max_samples")

    if max_samples is not None:
        train_samples = train_samples[: int(max_samples)]
        val_samples = val_samples[: max(1, int(max_samples) // 5)]

    observed_labels = sorted({int(y) for _p, y in train_samples + val_samples})

    # Store label metadata inside the saved Hydra config. Submission will reuse it.
    cfg.dataset.label_values = observed_labels

    if observed_labels:
        max_label = max(observed_labels)

        if int(cfg.model.num_classes) <= max_label:
            raise ValueError(
                f"model.num_classes={cfg.model.num_classes} but labels go up to {max_label}. "
                f"Set model.num_classes and num_classes to at least {max_label + 1}."
            )

        missing = sorted(set(range(max_label + 1)) - set(observed_labels))

        if missing:
            print(f"Warning: missing class indices in train/val labels: {missing}")

    print(
        f"Samples: train={len(train_samples)} val={len(val_samples)} "
        f"labels={observed_labels[:10]}...{observed_labels[-10:]}"
    )

    use_imagenet_norm = bool(cfg.model.pretrained)

    train_transform = build_transforms(
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
    )

    eval_transform = build_transforms(
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
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

    sampler = (
        build_balanced_sampler(train_samples)
        if bool(cfg.training.get("balanced_sampler", False))
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            bool(cfg.training.get("persistent_workers", True))
            and int(cfg.training.num_workers) > 0
        ),
        prefetch_factor=(
            int(cfg.training.get("prefetch_factor", 2))
            if int(cfg.training.num_workers) > 0
            else None
        ),
        drop_last=bool(cfg.training.get("drop_last", True)),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.get("eval_batch_size", cfg.training.batch_size)),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(
            bool(cfg.training.get("persistent_workers", True))
            and int(cfg.training.num_workers) > 0
        ),
        prefetch_factor=(
            int(cfg.training.get("prefetch_factor", 2))
            if int(cfg.training.num_workers) > 0
            else None
        ),
    )

    model = build_model(cfg).to(device)

    if bool(cfg.training.get("compile", False)) and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(
            model,
            mode=str(cfg.training.get("compile_mode", "max-autotune")),
        )

    if (
        device.type == "cuda"
        and bool(cfg.training.get("data_parallel", False))
        and torch.cuda.device_count() > 1
    ):
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    loss_fn = make_loss_fn(cfg, train_samples, device)
    optimizer = make_optimizer(cfg, model)

    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1))

    update_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / max(1, grad_accum_steps)),
    )

    scheduler, scheduler_interval = make_scheduler(
        cfg,
        optimizer,
        update_steps_per_epoch,
    )

    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    augmenter = None

    if bool(cfg.training.get("use_augmentation", False)):
        augmenter = VideoAugmentation(
            p_temporal_jitter=float(cfg.training.get("aug_temporal_jitter", 0.3)),
            p_frame_drop=float(cfg.training.get("aug_frame_drop", 0.2)),
            p_erasing=float(cfg.training.get("aug_erasing", 0.5)),
            p_mixup=float(cfg.training.get("aug_mixup", 0.3)),
            mixup_alpha=float(cfg.training.get("aug_mixup_alpha", 0.2)),
        )

    ema = None

    if bool(cfg.training.get("ema", True)):
        ema = ModelEMA(
            model,
            decay=float(cfg.training.get("ema_decay", 0.999)),
        ).to(device)

    best_val_accuracy = 0.0
    patience = int(cfg.training.get("early_stopping_patience", 0))
    epochs_without_improvement = 0

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    last_checkpoint_path = checkpoint_path.with_name(
        checkpoint_path.stem + "_last" + checkpoint_path.suffix
    )

    validate_ema = bool(cfg.training.get("validate_ema", True))
    save_training_state = bool(cfg.training.get("save_training_state", False))

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=int(cfg.training.epochs),
            augmenter=augmenter,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=float(cfg.training.get("max_grad_norm", 1.0)),
            ema=ema,
        )

        eval_model = ema.ema if ema is not None and validate_ema else unwrap_model(model)

        val_loss, val_acc = evaluate_epoch(
            eval_model,
            val_loader,
            loss_fn,
            device,
            epoch,
            int(cfg.training.epochs),
        )

        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e}"
        )

        # Lightweight last checkpoint.
        # If validate_ema=True, this writes EMA weights directly into model_state_dict.
        save_checkpoint(
            last_checkpoint_path,
            model,
            cfg,
            val_acc,
            epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            save_training_state=save_training_state,
            use_ema_weights=(ema is not None and validate_ema),
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0

            # Lightweight best checkpoint for create_submission.py.
            # If EMA was used for validation, save EMA weights as model_state_dict.
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                val_acc,
                epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                save_training_state=save_training_state,
                use_ema_weights=(ema is not None and validate_ema),
            )

            print(f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})")

        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement.")
            break

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()