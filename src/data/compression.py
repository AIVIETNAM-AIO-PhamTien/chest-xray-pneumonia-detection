"""Fixed-quantization JPEG standardisation for the encoding intervention.

The training split stores its normal radiographs with a finer quantization
table than everything else, which makes the class readable from the encoder
settings. A network never sees a header, so the question is whether the pixel
traces that table leaves are what the model learned. Answering it needs every
image to pass through one identical encoder.

Two levels are used, and they are not interchangeable.

Q75 is the modal quantization table across development, chosen without looking
at labels. It also happens to be the table almost all benchmark images carry,
so any gain from it confounds "the label-encoding correlation was removed" with
"development was moved into the benchmark's style".

Q85 exists to separate those. No folder in the dataset uses it, so a gain there
cannot come from matching the target. It is an off-support intermediate
control, not a neutral one -- re-encoding can only add quantization, never undo
what an earlier encoder discarded.
"""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image


def qtable_hash(tables: Dict[int, Sequence[int]]) -> str:
    """Stable short digest of a quantization table set.

    Args:
        tables: Mapping of table index to its 64 coefficients.

    Returns:
        First twelve hex characters of the SHA-256 digest.
    """
    serialised = {str(k): list(v) for k, v in tables.items()}
    return hashlib.sha256(json.dumps(serialised, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def read_qtable(path: Path) -> List[int]:
    """Read the luminance quantization table an existing file was written with.

    Args:
        path: Path to a JPEG file.

    Returns:
        The 64 luminance coefficients.
    """
    with Image.open(path) as handle:
        return list(handle.quantization[0])


def canonical_qtable(quality: int) -> List[int]:
    """Materialise the table an encoder produces at a given quality setting.

    The quality parameter is an encoder-specific shorthand, and different
    libraries map it to different tables. Encoding once and reading the result
    back pins the actual coefficients, so later runs cannot drift because a
    dependency changed its mapping.

    Args:
        quality: Encoder quality setting.

    Returns:
        The 64 luminance coefficients that setting produces.
    """
    probe = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L")
    buffer = BytesIO()
    probe.save(
        buffer, format="JPEG", quality=quality, optimize=False, progressive=False
    )
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return list(handle.quantization[0])


def roundtrip(array: np.ndarray, table: Sequence[int]) -> np.ndarray:
    """Re-encode and decode an image through one fixed quantization table.

    Args:
        array: Grayscale image as uint8.
        table: The 64 luminance coefficients to encode with.

    Returns:
        The decoded result, same shape and dtype.
    """
    buffer = BytesIO()
    Image.fromarray(array, mode="L").save(
        buffer,
        format="JPEG",
        qtables=[list(table)],
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def verify_roundtrip(array: np.ndarray, table: Sequence[int]) -> Dict[str, object]:
    """Encode once and report what the container actually ended up holding.

    Asserting the intended settings landed is cheap, and a silent fallback to a
    default table would invalidate the whole intervention.

    Args:
        array: Grayscale image as uint8.
        table: The 64 luminance coefficients to encode with.

    Returns:
        Mapping describing the encoded file.
    """
    buffer = BytesIO()
    Image.fromarray(array, mode="L").save(
        buffer,
        format="JPEG",
        qtables=[list(table)],
        optimize=False,
        progressive=False,
        subsampling=0,
    )
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return {
            "qtable_hash": qtable_hash(handle.quantization),
            "size": handle.size,
            "mode": handle.mode,
            "progressive": bool(
                handle.info.get("progressive") or handle.info.get("progression")
            ),
            "bytes": buffer.getbuffer().nbytes,
        }


def resize_for_cache(image: Image.Image, size: int, mode: str) -> np.ndarray:
    """Reduce an image to the model's input grid, matching Baseline v4 exactly.

    Reproduced here rather than imported so the intervention cannot silently
    diverge from the frozen pipeline if the notebook changes.

    Args:
        image: Source image.
        size: Output side length.
        mode: Either "stretch" or "letterbox".

    Returns:
        The resized grayscale array.

    Raises:
        ValueError: If the mode is not recognised.
    """
    gray = image.convert("L")
    if mode == "stretch":
        return np.asarray(gray.resize((size, size), Image.Resampling.BILINEAR))
    if mode != "letterbox":
        raise ValueError(f"Chế độ resize lạ: {mode}")
    gray.thumbnail((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(gray)
    canvas = Image.new("L", (size, size), color=int(np.median(array)))
    canvas.paste(gray, ((size - gray.width) // 2, (size - gray.height) // 2))
    return np.asarray(canvas)


def standardised_cache_entry(
    path: str, size: int, mode: str, table: Sequence[int]
) -> np.ndarray:
    """Produce one cache entry under the standardised encoding.

    Order matters. Geometry is applied first so the encoder always sees the
    same grid, and the re-encode happens before augmentation so this stays a
    fixed standardisation rather than a compression augmentation, which would
    be a different experiment.

    Args:
        path: Source image path.
        size: Model input side length.
        mode: Geometry mode.
        table: Quantization coefficients to standardise on.

    Returns:
        The decoded uint8 array the model will read.
    """
    with Image.open(path) as image:
        resized = resize_for_cache(image, size, mode)
    return roundtrip(resized, table)
