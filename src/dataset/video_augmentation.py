"""
VideoAugmentation: augmentations temporellement cohérentes pour des clips de T frames.

Principe clé : toutes les frames d'un même clip reçoivent la MÊME transformation
géométrique (flip, crop, rotation) mais des transformations pixel indépendantes
(jitter de couleur, effacement) pour simuler des variations naturelles.

Intégration dans train.py :
    from dataset.video_augmentation import VideoAugmentation, VideoAugmentationTransform

    # Remplacer build_transforms(is_training=True) par :
    train_transform = VideoAugmentationTransform(img_size=224, use_imagenet_norm=True)

    # Ou, si vous voulez augmenter le tensor (B, T, C, H, W) directement dans train_one_epoch :
    augmenter = VideoAugmentation()
    video_batch = augmenter(video_batch)   # à appeler sur GPU avant le forward

Architecture des augmentations :
    ┌── Niveau FRAME (appliqué frame par frame, indépendamment) ──────────┐
    │  ColorJitter (brightness, contrast, saturation, hue)                │
    │  GaussianBlur (flou léger, simule bougé de caméra)                  │
    │  RandomErasing (masque un patch rectangulaire → force le modèle     │
    │                 à ne pas se fier à une seule région)                │
    └─────────────────────────────────────────────────────────────────────┘
    ┌── Niveau CLIP (même transform pour toutes les frames) ──────────────┐
    │  RandomHorizontalFlip  (cohérent temporellement)                    │
    │  RandomResizedCrop     (zoom aléatoire, même crop sur chaque frame) │
    │  RandomRotation        (±10°, simule caméra de guingois)            │
    │  TemporalOrderJitter   (échange deux frames adjacentes ~20% du tps) │
    │  FrameDropAndRepeat    (supprime une frame, duplique une voisine)   │
    └─────────────────────────────────────────────────────────────────────┘

Usage dans utils.py / train.py :
    Voir VideoAugmentationTransform pour un remplacement direct de build_transforms.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Transform PIL : remplace build_transforms(is_training=True)
#    Appliqué frame par frame dans VideoFrameDataset.__getitem__
#    MAIS avec un état partagé (seed) pour la cohérence temporelle.
# ---------------------------------------------------------------------------

class VideoAugmentationTransform:
    """
    Remplacement direct de build_transforms(is_training=True).

    La cohérence temporelle est assurée en fixant le même seed random pour
    toutes les frames d'un même clip lors du crop/flip géométrique,
    tout en laissant le jitter couleur varier frame par frame.

    Utilisation dans train.py — remplacer :
        train_transform = build_transforms(is_training=True, use_imagenet_norm=True)
    par :
        from dataset.video_augmentation import VideoAugmentationTransform
        train_transform = VideoAugmentationTransform(img_size=224, use_imagenet_norm=True)

    IMPORTANT : Cette classe est stateful (self._clip_seed).
    Avant de charger chaque nouveau clip, appeler transform.new_clip().
    Voir l'exemple d'intégration dans VideoFrameDataset ci-dessous.
    """

    def __init__(self, img_size: int = 224, use_imagenet_norm: bool = True):
        self.img_size = img_size
        self._clip_seed: int = 0

        if use_imagenet_norm:
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Jitter couleur : indépendant par frame (intentionnel)
        self.color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1,
        )
        # Flou gaussien léger : simule bougé
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

        self.to_tensor = T.ToTensor()

    def new_clip(self) -> None:
        """À appeler avant de transformer les frames d'un nouveau clip."""
        self._clip_seed = random.randint(0, 2**31)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Transforme une seule frame PIL → tensor (C, H, W)."""
        # --- Transformations GÉOMÉTRIQUES cohérentes (même seed) ---
        rng_state = random.getstate()
        random.seed(self._clip_seed)
        torch_state = torch.get_rng_state()
        torch.manual_seed(self._clip_seed)

        # RandomResizedCrop cohérent
        i, j, h, w = T.RandomResizedCrop.get_params(
            img,
            scale=(0.7, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))

        # RandomHorizontalFlip cohérent
        if random.random() < 0.5:
            img = TF.hflip(img)

        # RandomRotation cohérent (±10°)
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle)

        # Restaurer l'état random pour les augmentations indépendantes
        random.setstate(rng_state)
        torch.set_rng_state(torch_state)

        # --- Transformations PIXEL indépendantes (varient par frame) ---
        # ColorJitter avec probabilité 80%
        if random.random() < 0.8:
            img = self.color_jitter(img)

        # GaussianBlur avec probabilité 20%
        if random.random() < 0.2:
            img = self.gaussian_blur(img)

        # Grayscale avec probabilité 10% (force invariance couleur)
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)

        return tensor


# ---------------------------------------------------------------------------
# 2. Augmentations sur tenseur (B, T, C, H, W) — à utiliser dans train_one_epoch
#    Complémentaire aux augmentations PIL ci-dessus.
# ---------------------------------------------------------------------------

class VideoAugmentation(nn.Module):
    """
    Augmentations appliquées sur le batch tensor (B, T, C, H, W) directement,
    idéalement sur GPU avant le forward pass.

    Inclut :
    - TemporalOrderJitter : échange aléatoire de 2 frames adjacentes
    - FrameDropAndRepeat  : supprime une frame, duplique la précédente
    - RandomErasing       : efface un patch rectangulaire sur chaque frame
    - MixUp temporel      : mixe deux clips du batch (alpha=0.2)

    Usage dans train_one_epoch :
        augmenter = VideoAugmentation(p_temporal_jitter=0.3,
                                       p_frame_drop=0.2,
                                       p_erasing=0.5,
                                       p_mixup=0.3)
        ...
        video_batch = augmenter(video_batch)
        logits = model(video_batch)
    """

    def __init__(
        self,
        p_temporal_jitter: float = 0.3,
        p_frame_drop: float = 0.2,
        p_erasing: float = 0.5,
        p_mixup: float = 0.3,
        mixup_alpha: float = 0.2,
        erasing_scale: Tuple[float, float] = (0.02, 0.2),
        erasing_ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        super().__init__()
        self.p_temporal_jitter = p_temporal_jitter
        self.p_frame_drop = p_frame_drop
        self.p_erasing = p_erasing
        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha
        self.erasing_scale = erasing_scale
        self.erasing_ratio = erasing_ratio

    def forward(
        self,
        video_batch: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            video_batch : (B, T, C, H, W)
            labels      : (B,) optionnel — nécessaire pour MixUp

        Returns:
            Si labels=None : video_batch augmenté (B, T, C, H, W)
            Si labels fournis et MixUp activé : (video_batch, labels_a, labels_b, lam)
                → utiliser dans la loss : lam*CE(logits, labels_a) + (1-lam)*CE(logits, labels_b)
        """
        x = video_batch  # (B, T, C, H, W)

        # 1. Temporal Order Jitter — échange deux frames adjacentes
        if random.random() < self.p_temporal_jitter:
            x = self._temporal_jitter(x)

        # 2. Frame Drop & Repeat — supprime une frame, duplique une voisine
        if random.random() < self.p_frame_drop:
            x = self._frame_drop_repeat(x)

        # 3. Random Erasing — sur chaque frame indépendamment
        if random.random() < self.p_erasing:
            x = self._random_erasing(x)

        # 4. MixUp temporel
        if labels is not None and random.random() < self.p_mixup:
            x, labels_a, labels_b, lam = self._mixup(x, labels)
            return x, labels_a, labels_b, lam

        return x

    # ------------------------------------------------------------------
    def _temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Échange deux frames adjacentes aléatoires dans chaque clip du batch."""
        B, T, C, H, W = x.shape
        if T < 2:
            return x
        x = x.clone()
        # Choisir un indice aléatoire entre 0 et T-2, échanger avec i+1
        idx = random.randint(0, T - 2)
        x[:, idx], x[:, idx + 1] = x[:, idx + 1].clone(), x[:, idx].clone()
        return x

    def _frame_drop_repeat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supprime une frame aléatoire et duplique la frame précédente (ou suivante
        si on supprime la première), maintenant T constant.
        Simule une vidéo avec un frame manquant / caméra qui saute.
        """
        B, T, C, H, W = x.shape
        if T < 2:
            return x
        x = x.clone()
        drop_idx = random.randint(0, T - 1)
        replace_idx = max(0, drop_idx - 1)
        x[:, drop_idx] = x[:, replace_idx]
        return x

    def _random_erasing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efface un patch rectangulaire aléatoire sur TOUTES les frames
        (même région → cohérence temporelle, cache une partie de la scène).
        Valeur de remplacement : moyenne ImageNet normalisée ≈ 0.
        """
        B, T, C, H, W = x.shape
        scale_lo, scale_hi = self.erasing_scale
        ratio_lo, ratio_hi = self.erasing_ratio

        area = H * W
        erase_area = random.uniform(scale_lo, scale_hi) * area
        aspect = random.uniform(ratio_lo, ratio_hi)

        eh = int(round((erase_area * aspect) ** 0.5))
        ew = int(round((erase_area / aspect) ** 0.5))
        eh = min(eh, H - 1)
        ew = min(ew, W - 1)

        top = random.randint(0, H - eh)
        left = random.randint(0, W - ew)

        x = x.clone()
        x[:, :, :, top:top + eh, left:left + ew] = 0.0
        return x

    def _mixup(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        MixUp : mélange deux clips du batch avec coefficient lambda ~ Beta(alpha, alpha).
        Retourne (mixed_x, labels_a, labels_b, lam) pour la loss mixup.
        """
        import numpy as np
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        B = x.shape[0]
        perm = torch.randperm(B, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[perm]
        labels_a = labels
        labels_b = labels[perm]
        return mixed_x, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# 3. Helper : perte MixUp
# ---------------------------------------------------------------------------

def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Calcule la perte MixUp : lam * CE(logits, a) + (1-lam) * CE(logits, b).

    Usage dans train_one_epoch :
        result = augmenter(video_batch, labels)
        if isinstance(result, tuple):
            video_batch, labels_a, labels_b, lam = result
            logits = model(video_batch)
            loss = mixup_criterion(loss_fn, logits, labels_a, labels_b, lam)
        else:
            video_batch = result
            logits = model(video_batch)
            loss = loss_fn(logits, labels)
    """
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# 4. Label Smoothing Loss (complément anti-overfitting)
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy avec label smoothing.
    smoothing=0.1 signifie que la vraie classe reçoit 0.9 de probabilité cible
    et les autres classes se partagent 0.1, ce qui pénalise la sur-confiance.

    Usage dans train.py :
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        # (remplace nn.CrossEntropyLoss())
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Soft targets : (1 - smoothing) pour la vraie classe, smoothing/(C-1) pour les autres
        with torch.no_grad():
            soft_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(soft_targets * log_probs).sum(dim=-1)
        return loss.mean()
