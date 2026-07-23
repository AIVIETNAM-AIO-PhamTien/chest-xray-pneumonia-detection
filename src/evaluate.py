"""Evaluation metrics for the pneumonia classifier."""

from typing import Dict

import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, object]:
    """Compute classification metrics over a full DataLoader.

    Args:
        model: Model to evaluate (this function switches it to eval mode).
        loader: DataLoader yielding (images, labels) batches.
        device: Device to run inference on.

    Returns:
        A dict with float entries "accuracy", "precision", "recall", "f1",
        plus "confusion_matrix" holding a nested list of counts.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    accuracy = sum(p == label for p, label in zip(all_preds, all_labels)) / len(
        all_labels
    )

    return {
        "accuracy": accuracy,
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
