import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

def make_weighted_sampler(labels):
    labels_t = torch.tensor(labels)
    class_counts = torch.bincount(labels_t)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_t]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts

def make_class_weight_loss(class_counts):
    total = class_counts.sum().float()
    weights = total / (len(class_counts) * class_counts.float())
    return nn.CrossEntropyLoss(weight=weights)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()