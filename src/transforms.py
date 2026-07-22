"""Image transform pipelines for training and evaluation."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the training-time augmentation pipeline.

    Chest X-rays are single-channel; they are duplicated to 3 channels so
    the ImageNet-pretrained backbone can consume them.

    Args:
        image_size: Target square size (in pixels) for resized images.

    Returns:
        A torchvision transform pipeline for training images.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic pipeline used for validation and test data.

    Args:
        image_size: Target square size (in pixels) for resized images.

    Returns:
        A torchvision transform pipeline for validation/test images.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
