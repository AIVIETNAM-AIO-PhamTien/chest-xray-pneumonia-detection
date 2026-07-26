"""Tests for fixed-quantization JPEG standardisation."""

import numpy as np
import pytest
from PIL import Image

from src.data.compression import (
    canonical_qtable,
    qtable_hash,
    resize_for_cache,
    roundtrip,
    verify_roundtrip,
)


@pytest.fixture
def textured():
    """A grayscale image with enough detail for compression to matter.

    Returns:
        A 224x224 uint8 array.
    """
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 6 * np.pi, 224)
    base = 128 + 60 * np.sin(grid)[None, :] * np.cos(grid)[:, None]
    return np.clip(base + rng.normal(0, 12, (224, 224)), 0, 255).astype(np.uint8)


def test_canonical_qtable_has_64_positive_coefficients():
    """A luminance table is 64 entries and none may be zero."""
    table = canonical_qtable(85)
    assert len(table) == 64
    assert all(value > 0 for value in table)


def test_higher_quality_quantizes_more_finely():
    """Raising quality must lower the coefficients, or the levels are mislabelled."""
    assert sum(canonical_qtable(95)) < sum(canonical_qtable(85))
    assert sum(canonical_qtable(85)) < sum(canonical_qtable(75))


def test_canonical_qtable_is_deterministic():
    """The frozen configuration is worthless if the table drifts between calls."""
    assert canonical_qtable(85) == canonical_qtable(85)


def test_roundtrip_preserves_shape_and_dtype(textured):
    """The cache contract is uint8 at the same grid."""
    out = roundtrip(textured, canonical_qtable(75))
    assert out.shape == textured.shape
    assert out.dtype == np.uint8


def test_roundtrip_actually_writes_the_requested_table(textured):
    """A silent fallback to a default table would void the intervention."""
    table = canonical_qtable(85)
    info = verify_roundtrip(textured, table)
    assert info["qtable_hash"] == qtable_hash({0: table})
    assert info["size"] == (224, 224)
    assert info["mode"] == "L"
    assert info["progressive"] is False


def test_lower_quality_loses_more_detail(textured):
    """Coarser quantization must move the pixels further from the original."""
    fine = roundtrip(textured, canonical_qtable(95)).astype(float)
    coarse = roundtrip(textured, canonical_qtable(75)).astype(float)
    reference = textured.astype(float)
    assert np.abs(coarse - reference).mean() > np.abs(fine - reference).mean()


def test_standardising_to_the_coarser_level_converges_two_encodings(textured):
    """Re-encoding at the coarser of two levels brings them together.

    This is the dataset's situation: normals stored finely, everything else
    coarsely. Quantizing the fine copy down to the coarse level removes the
    detail only it had.
    """
    fine = roundtrip(textured, canonical_qtable(95))
    coarse = roundtrip(textured, canonical_qtable(75))
    before = np.abs(fine.astype(float) - coarse.astype(float)).mean()

    table = canonical_qtable(75)
    after = np.abs(roundtrip(fine, table).astype(float)
                   - roundtrip(coarse, table).astype(float)).mean()
    assert after < before


def test_standardising_to_an_intermediate_level_does_not_converge(textured):
    """An intermediate target leaves the two apart, and can widen the gap.

    Pins why the Q85 control could not work: re-encoding only ever removes
    detail, so a level finer than the coarser input cannot lift that input to
    meet the other. Choosing a middle quality neutralises nothing.
    """
    fine = roundtrip(textured, canonical_qtable(95))
    coarse = roundtrip(textured, canonical_qtable(75))
    before = np.abs(fine.astype(float) - coarse.astype(float)).mean()

    table = canonical_qtable(85)
    after = np.abs(roundtrip(fine, table).astype(float)
                   - roundtrip(coarse, table).astype(float)).mean()
    assert after >= before


def test_letterbox_preserves_aspect_and_pads(textured):
    """Letterbox must not distort; a wide image gets padded, not squashed."""
    wide = Image.fromarray(textured).resize((400, 200))
    out = resize_for_cache(wide, 224, "letterbox")
    assert out.shape == (224, 224)
    # Padding sits in the top and bottom bands for a wide source.
    assert out[:10].std() < out[100:124].std()


def test_stretch_fills_the_whole_canvas(textured):
    """Stretch must squash rather than pad, so no uniform band appears."""
    wide = Image.fromarray(textured).resize((400, 200))
    out = resize_for_cache(wide, 224, "stretch")
    assert out.shape == (224, 224)
    assert out[:10].std() > 1.0


def test_unknown_resize_mode_is_rejected(textured):
    """An unrecognised mode must fail loudly rather than pick a default."""
    with pytest.raises(ValueError, match="Chế độ resize lạ"):
        resize_for_cache(Image.fromarray(textured), 224, "crop")
