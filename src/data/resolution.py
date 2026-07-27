"""Source-resolution bottlenecks applied before the frozen geometry pipeline.

Normal radiographs are stored at roughly three times the pixel count of
pneumonia ones in both splits. After the pipeline reduces everything to
224x224, that history survives as texture: a picture downsampled from three
megapixels does not look like one downsampled from one, even at the same final
size. The bottleneck removes the advantage by passing every image through a
common ceiling first.

Only downsampling is ever applied. Upsampling a small image to meet a ceiling
would invent detail and change what is being tested, so images already below
the ceiling pass through untouched and are counted separately.
"""

from typing import Dict, Optional

import numpy as np
from PIL import Image

#: Pre-registered ceilings in megapixels, chosen before any model was run.
BOTTLENECKS: Dict[str, Optional[float]] = {
    "native": None,
    "mp0.8": 0.8,
    "mp0.5": 0.5,
    "mp0.25": 0.25,
}


def apply_bottleneck(image: Image.Image,
                     megapixels: Optional[float]) -> Image.Image:
    """Reduce an image to a pixel-count ceiling, preserving aspect ratio.

    Args:
        image: Source image.
        megapixels: Ceiling in megapixels, or None to pass through.

    Returns:
        The image at or below the ceiling.
    """
    if megapixels is None:
        return image
    current = image.width * image.height / 1e6
    if current <= megapixels:
        return image
    scale = np.sqrt(megapixels / current)
    width = max(int(round(image.width * scale)), 1)
    height = max(int(round(image.height * scale)), 1)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def would_upsample(width: int, height: int,
                   megapixels: Optional[float]) -> bool:
    """Whether an image is already below a ceiling and so is left alone.

    Args:
        width: Image width.
        height: Image height.
        megapixels: Ceiling in megapixels, or None.

    Returns:
        True when the image is smaller than the ceiling.
    """
    if megapixels is None:
        return False
    return (width * height / 1e6) < megapixels
