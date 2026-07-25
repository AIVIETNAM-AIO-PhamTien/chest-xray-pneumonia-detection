from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class AddGaussianNoise:
    """Add mild zero-mean Gaussian noise to a tensor in [0, 1]."""

    def __init__(self, std_range: Tuple[float, float] = (0.005, 0.02)) -> None:
        low, high = std_range
        if low < 0 or high < low:
            raise ValueError("std_range must satisfy 0 <= low <= high.")
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        low, high = self.std_range
        std = torch.empty(1).uniform_(low, high).item()
        return (tensor + torch.randn_like(tensor) * std).clamp_(0.0, 1.0)


class AddSpeckleNoise:
    """Add mild multiplicative speckle noise to a tensor in [0, 1]."""

    def __init__(self, std_range: Tuple[float, float] = (0.005, 0.02)) -> None:
        low, high = std_range
        if low < 0 or high < low:
            raise ValueError("std_range must satisfy 0 <= low <= high.")
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        low, high = self.std_range
        std = torch.empty(1).uniform_(low, high).item()
        noise = torch.randn_like(tensor) * std
        return (tensor + tensor * noise).clamp_(0.0, 1.0)


def _normalization_values(
    channels: int,
    pretrained: bool,
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
) -> tuple[list[float], list[float]]:
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3.")

    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together.")

    if mean is not None and std is not None:
        mean_list = list(mean)
        std_list = list(std)
        if len(mean_list) != channels or len(std_list) != channels:
            raise ValueError("mean and std lengths must equal channels.")
        if any(value <= 0 for value in std_list):
            raise ValueError("All std values must be positive.")
        return mean_list, std_list

    if pretrained:
        if channels != 3:
            raise ValueError("pretrained=True requires channels=3.")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    return [0.5] * channels, [0.5] * channels


def get_train_transforms(
    img_size: int = 128,
    channels: int = 3,
    policy: str = "advanced",
    pretrained: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """Build training transforms.

    Policies:
      - ``none``: deterministic resize only.
      - ``paper``: horizontal flip, 0.9-1.1 zoom, and +/-20% brightness.
      - ``advanced``: paper policy plus conservative rotation, translation,
        contrast, blur, and low-amplitude sensor-like noise.

    Vertical flipping and aggressive crops are intentionally excluded because
    they can create anatomically implausible chest radiographs.
    """

    policy = policy.lower()
    if policy not in {"none", "paper", "advanced"}:
        raise ValueError("policy must be one of: none, paper, advanced.")
    if img_size <= 0:
        raise ValueError("img_size must be positive.")

    norm_mean, norm_std = _normalization_values(
        channels=channels,
        pretrained=pretrained,
        mean=mean,
        std=std,
    )

    ops = [
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]

    if policy == "paper":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=0,
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.2),
            ]
        )
    elif policy == "advanced":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.03, 0.03),
                    scale=(0.9, 1.1),
                    shear=(-2.0, 2.0),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.15),
                transforms.RandomAutocontrast(p=0.10),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.10,
                ),
            ]
        )

    ops.append(transforms.ToTensor())

    if policy == "advanced":
        ops.append(
            transforms.RandomApply(
                [
                    transforms.RandomChoice(
                        [
                            AddGaussianNoise((0.005, 0.02)),
                            AddSpeckleNoise((0.005, 0.02)),
                        ]
                    )
                ],
                p=0.15,
            )
        )

    ops.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    return transforms.Compose(ops)


def get_eval_transforms(
    img_size: int = 128,
    channels: int = 3,
    pretrained: bool = False,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """Build deterministic validation/test transforms."""

    norm_mean, norm_std = _normalization_values(
        channels=channels,
        pretrained=pretrained,
        mean=mean,
        std=std,
    )

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=channels),
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
