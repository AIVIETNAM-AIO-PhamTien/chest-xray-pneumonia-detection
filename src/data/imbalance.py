from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

TensorLike = Union[torch.Tensor, Sequence[int], Iterable[int]]


def _as_label_tensor(labels: TensorLike) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        values = labels.detach().cpu().long().flatten()
    else:
        values = torch.tensor([int(value) for value in labels], dtype=torch.long)

    if values.numel() == 0:
        raise ValueError("labels must not be empty.")
    if (values < 0).any():
        raise ValueError("labels must be non-negative integers.")
    return values


def count_classes(
    labels: TensorLike, num_classes: Optional[int] = None
) -> torch.Tensor:
    labels_t = _as_label_tensor(labels)
    inferred = int(labels_t.max().item()) + 1
    if num_classes is None:
        num_classes = inferred
    if num_classes < inferred:
        raise ValueError("num_classes is smaller than the largest label.")

    counts = torch.bincount(labels_t, minlength=num_classes)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training data is missing classes: {missing}")
    return counts


def inverse_frequency_weights(
    labels: TensorLike,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Return class weights normalized to mean 1."""

    counts = count_classes(labels, num_classes=num_classes).float()
    weights = 1.0 / counts
    return weights / weights.mean()


def effective_number_weights(
    labels: TensorLike,
    num_classes: Optional[int] = None,
    beta: float = 0.999,
) -> torch.Tensor:
    """Class-balanced weights based on effective sample count."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1).")

    counts = count_classes(labels, num_classes=num_classes).float()
    effective_num = 1.0 - torch.pow(torch.tensor(beta), counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.mean()


def make_weighted_sampler(
    labels: TensorLike,
    num_classes: Optional[int] = None,
    seed: int = 42,
) -> tuple[WeightedRandomSampler, torch.Tensor]:
    """Create a training-only sampler that balances classes in expectation."""

    labels_t = _as_label_tensor(labels)
    counts = count_classes(labels_t, num_classes=num_classes)
    class_weights = inverse_frequency_weights(labels_t, num_classes=len(counts))
    sample_weights = class_weights[labels_t].double()

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, counts


class FocalLoss(nn.Module):
    """Multi-class focal loss for logits.

    ``alpha`` is applied after ``p_t`` is computed. This avoids the common bug
    of deriving ``p_t`` from a class-weighted cross-entropy value.
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum.")

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32).flatten()
            if (alpha_t < 0).any():
                raise ValueError("alpha values must be non-negative.")
            self.register_buffer("alpha", alpha_t)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, num_classes].")
        targets = targets.long().flatten()
        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets batch sizes do not match.")

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -torch.pow(1.0 - pt, self.gamma) * log_pt

        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("alpha length must match num_classes.")
            loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_classification_loss(
    strategy: str,
    labels: TensorLike,
    num_classes: int = 2,
    gamma: float = 2.0,
    beta: float = 0.999,
) -> nn.Module:
    """Build one imbalance-aware criterion.

    Strategies: ``none``, ``weighted_ce``, ``effective_ce``, ``focal``, and
    ``effective_focal``. Use this or a weighted sampler, not both by default.
    """

    strategy = strategy.lower()
    if strategy == "none":
        return nn.CrossEntropyLoss()
    if strategy == "weighted_ce":
        weights = inverse_frequency_weights(labels, num_classes=num_classes)
        return nn.CrossEntropyLoss(weight=weights)
    if strategy == "effective_ce":
        weights = effective_number_weights(
            labels,
            num_classes=num_classes,
            beta=beta,
        )
        return nn.CrossEntropyLoss(weight=weights)
    if strategy == "focal":
        alpha = inverse_frequency_weights(labels, num_classes=num_classes)
        return FocalLoss(alpha=alpha, gamma=gamma)
    if strategy == "effective_focal":
        alpha = effective_number_weights(
            labels,
            num_classes=num_classes,
            beta=beta,
        )
        return FocalLoss(alpha=alpha, gamma=gamma)

    raise ValueError(
        "Unknown strategy. Choose from: none, weighted_ce, effective_ce, "
        "focal, effective_focal."
    )
