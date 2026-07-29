"""Acquisition and image-quality descriptors computed from raw radiographs.

Every feature here is deliberately blind to pathology. They describe how an
image was captured, cropped, exposed and stored, not what is inside the lungs.
That is the point: if these alone separate development normals from benchmark
normals, the split differs in ways that have nothing to do with disease.

Features are read from the decoded original: before resize, before
normalisation, without augmentation, through one fixed grayscale conversion.
Anything computed after the pipeline's resize would measure the pipeline.

Two features are computed on a fixed-size copy rather than natively, and are
named to say so. Frequency content and blockiness scale with pixel count, so
measuring them natively would re-encode resolution, which is already its own
feature and would otherwise be counted twice.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from scipy import ndimage

#: Side length for features that must not inherit the native resolution.
NORMALISED_SIDE = 256
#: Otsu is computed on this many bins; radiographs are 8-bit.
HISTOGRAM_BINS = 256


def _otsu_threshold(gray: np.ndarray) -> float:
    """Intensity that best separates an image into two classes.

    Args:
        gray: Grayscale image as uint8.

    Returns:
        Threshold in [0, 255].
    """
    counts = np.bincount(gray.ravel(), minlength=HISTOGRAM_BINS).astype(float)
    weights = counts / counts.sum()
    levels = np.arange(HISTOGRAM_BINS)
    omega = np.cumsum(weights)
    mu = np.cumsum(weights * levels)
    total = mu[-1]
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total * omega - mu) ** 2 / denominator
    between[~np.isfinite(between)] = 0.0
    return float(np.argmax(between))


def _geometry(gray: np.ndarray) -> Dict[str, float]:
    """Crop, framing and how much of the canvas the body occupies.

    Args:
        gray: Grayscale image as uint8.

    Returns:
        Mapping of geometry feature names to values.
    """
    height, width = gray.shape
    threshold = _otsu_threshold(gray)
    foreground = gray > threshold

    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if len(rows) and len(cols):
        top, bottom = int(rows[0]), int(rows[-1])
        left, right = int(cols[0]), int(cols[-1])
    else:
        top, bottom, left, right = 0, height - 1, 0, width - 1

    box_height, box_width = bottom - top + 1, right - left + 1
    quarter_h, quarter_w = max(height // 4, 1), max(width // 4, 1)
    centre = gray[quarter_h : height - quarter_h, quarter_w : width - quarter_w]
    border = np.concatenate(
        [
            gray[:quarter_h].ravel(),
            gray[-quarter_h:].ravel(),
            gray[:, :quarter_w].ravel(),
            gray[:, -quarter_w:].ravel(),
        ]
    )
    corner = max(min(height, width) // 10, 4)
    corners = np.array(
        [
            gray[:corner, :corner].mean(),
            gray[:corner, -corner:].mean(),
            gray[-corner:, :corner].mean(),
            gray[-corner:, -corner:].mean(),
        ]
    )

    return {
        "width": float(width),
        "height": float(height),
        "log_area": float(np.log(width * height)),
        "aspect": float(width / height),
        "fg_box_width_frac": float(box_width / width),
        "fg_box_height_frac": float(box_height / height),
        "fg_occupancy": float(foreground.mean()),
        "fg_box_occupancy": float((box_width * box_height) / (width * height)),
        "margin_top_frac": float(top / height),
        "margin_bottom_frac": float((height - 1 - bottom) / height),
        "margin_left_frac": float(left / width),
        "margin_right_frac": float((width - 1 - right) / width),
        "background_frac": float((~foreground).mean()),
        "centre_border_contrast": float(centre.mean() - border.mean()),
        "corner_uniformity": float(corners.std()),
        "corner_mean": float(corners.mean()),
    }


def _photometry(gray: np.ndarray) -> Dict[str, float]:
    """Exposure, contrast and the shape of the intensity histogram.

    Args:
        gray: Grayscale image as uint8.

    Returns:
        Mapping of photometric feature names to values.
    """
    values = gray.ravel().astype(np.float64)
    mean, std = float(values.mean()), float(values.std())
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    counts = np.bincount(gray.ravel(), minlength=HISTOGRAM_BINS).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    centred = values - mean
    scale = std if std > 1e-9 else 1.0

    return {
        "intensity_mean": mean,
        "intensity_std": std,
        **{
            f"intensity_p{int(q)}": float(v)
            for q, v in zip([1, 5, 25, 50, 75, 95, 99], percentiles)
        },
        "dynamic_range": float(percentiles[6] - percentiles[0]),
        "iqr": float(percentiles[4] - percentiles[2]),
        "histogram_entropy": float(-(probabilities * np.log2(probabilities)).sum()),
        "skewness": float(np.mean((centred / scale) ** 3)),
        "kurtosis": float(np.mean((centred / scale) ** 4) - 3.0),
        "near_black_frac": float((gray <= 10).mean()),
        "near_white_frac": float((gray >= 245).mean()),
    }


def _quality(gray: np.ndarray, file_size: int) -> Dict[str, float]:
    """Sharpness, noise and compression traces.

    Args:
        gray: Grayscale image as uint8.
        file_size: Size of the source file in bytes.

    Returns:
        Mapping of quality feature names to values.
    """
    image = gray.astype(np.float64)
    laplacian = ndimage.laplace(image)
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    gradient = np.hypot(gx, gy)

    # Immerkaer noise estimate: a kernel that annihilates smooth content, so
    # what survives is dominated by pixel noise rather than anatomy.
    kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    residual = ndimage.convolve(image, kernel, mode="reflect")
    noise = (
        np.abs(residual).mean()
        * np.sqrt(np.pi / 2.0)
        / (6.0 * max(np.prod(gray.shape) ** 0.0, 1.0))
    )

    small = np.asarray(
        Image.fromarray(gray).resize(
            (NORMALISED_SIDE, NORMALISED_SIDE), Image.BILINEAR
        ),
        dtype=np.float64,
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small - small.mean())))
    grid = np.arange(NORMALISED_SIDE) - NORMALISED_SIDE / 2
    radius = np.hypot(*np.meshgrid(grid, grid, indexing="ij"))
    high = radius > NORMALISED_SIDE / 8.0
    total_energy = spectrum.sum()

    # JPEG stores 8x8 blocks; a compressed image has larger steps across block
    # boundaries than inside them. The ratio is near 1 when uncompressed.
    def step(left_start, right_start):
        left, right = small[:, left_start::8], small[:, right_start::8]
        width = min(left.shape[1], right.shape[1])
        return np.abs(left[:, :width] - right[:, :width]).mean()

    blockiness = step(7, 8) / max(step(3, 4), 1e-9)

    pixels = float(gray.size)
    return {
        "laplacian_variance": float(laplacian.var()),
        "tenengrad": float((gradient**2).mean()),
        "edge_density": float((gradient > gradient.mean() + gradient.std()).mean()),
        "noise_estimate": float(noise),
        "unique_gray_levels": float(len(np.unique(gray))),
        "file_size_bytes": float(file_size),
        "file_size_per_pixel": float(file_size / pixels),
        "hf_energy_frac_norm256": float(spectrum[high].sum() / max(total_energy, 1e-9)),
        "blockiness_norm256": float(blockiness),
    }


def extract(path: str) -> Dict[str, float]:
    """Compute every nuisance descriptor for one radiograph.

    Args:
        path: Path to the image file.

    Returns:
        Mapping of feature name to value, including the source path.
    """
    location = Path(path)
    with Image.open(location) as handle:
        gray = np.asarray(handle.convert("L"), dtype=np.uint8)
    return {
        "path": str(location),
        **_geometry(gray),
        **_photometry(gray),
        **_quality(gray, location.stat().st_size),
    }


#: Pre-registered families. Declared here so the audit cannot quietly regroup
#: features after seeing which ones separate the domains.
FAMILIES = {
    "geometry": [
        "width",
        "height",
        "log_area",
        "aspect",
        "fg_box_width_frac",
        "fg_box_height_frac",
        "fg_occupancy",
        "fg_box_occupancy",
        "margin_top_frac",
        "margin_bottom_frac",
        "margin_left_frac",
        "margin_right_frac",
        "background_frac",
        "centre_border_contrast",
        "corner_uniformity",
        "corner_mean",
    ],
    "photometry": [
        "intensity_mean",
        "intensity_std",
        "intensity_p1",
        "intensity_p5",
        "intensity_p25",
        "intensity_p50",
        "intensity_p75",
        "intensity_p95",
        "intensity_p99",
        "dynamic_range",
        "iqr",
        "histogram_entropy",
        "skewness",
        "kurtosis",
        "near_black_frac",
        "near_white_frac",
    ],
    "quality": [
        "laplacian_variance",
        "tenengrad",
        "edge_density",
        "noise_estimate",
        "unique_gray_levels",
        "file_size_bytes",
        "file_size_per_pixel",
        "hf_energy_frac_norm256",
        "blockiness_norm256",
    ],
}

ALL_FEATURES = [name for names in FAMILIES.values() for name in names]
