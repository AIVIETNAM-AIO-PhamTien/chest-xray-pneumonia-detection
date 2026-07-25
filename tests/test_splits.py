"""Tests for patient-group parsing and the two validation-split protocols."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.splits import (
    build_manifest,
    count_leaked_groups,
    make_splits,
    manifest_rows,
    parse_group_id,
    split_summary,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("person1_bacteria_1.jpeg", "pneumonia:bacteria:1"),
        ("person1000_virus_1681.jpeg", "pneumonia:virus:1000"),
        ("person1_virus_6_1.jpeg", "pneumonia:virus:1"),
        ("IM-0115-0001.jpeg", "normal:im:115"),
        ("IM-0115-0001-0001.jpeg", "normal:im:115"),
        ("NORMAL2-IM-1427-0001.jpeg", "normal:normal2:1427"),
    ],
)
def test_parse_group_id_known_patterns(filename: str, expected: str) -> None:
    """Every filename shape in the published dataset maps to a group."""
    assert parse_group_id(filename) == expected


def test_bacteria_and_virus_person_numbers_are_different_patients() -> None:
    """person1_bacteria and person1_virus must not collapse into one group.

    The two subtypes run independent counters from 1, so treating the bare
    person number as a patient id invents cross-split overlaps that are not
    real. This test pins the distinction that resolves that.
    """
    assert parse_group_id("person1_bacteria_1.jpeg") != parse_group_id(
        "person1_virus_6.jpeg"
    )


def test_parse_group_id_rejects_unknown_pattern() -> None:
    """Unknown names must raise, not silently form a bogus patient group."""
    with pytest.raises(ValueError, match="Unrecognised filename pattern"):
        parse_group_id("scan_042.png")


def _make_dataset(root: Path) -> None:
    """Write a tiny dataset with patients that own multiple images.

    Args:
        root: Directory to hold the train/val/test splits.
    """
    plan = {
        "train": {
            "NORMAL": [f"IM-{i:04d}-0001.jpeg" for i in range(1, 21)],
            "PNEUMONIA": [
                f"person{i}_bacteria_{j}.jpeg" for i in range(1, 21) for j in (1, 2)
            ],
        },
        "val": {
            "NORMAL": ["NORMAL2-IM-9001-0001.jpeg"],
            "PNEUMONIA": ["person900_bacteria_1.jpeg"],
        },
        "test": {
            "NORMAL": [f"IM-{i:04d}-0001.jpeg" for i in range(500, 510)],
            "PNEUMONIA": [f"person{i}_virus_{i}.jpeg" for i in range(1, 11)],
        },
    }
    for split, classes in plan.items():
        for cls, names in classes.items():
            directory = root / split / cls
            directory.mkdir(parents=True, exist_ok=True)
            for name in names:
                Image.new("L", (32, 32), color=128).save(directory / name)


@pytest.fixture
def manifest(tmp_path: Path) -> pd.DataFrame:
    """Build a manifest over a tiny synthetic dataset.

    Args:
        tmp_path: pytest scratch directory.

    Returns:
        The manifest DataFrame.
    """
    root = tmp_path / "chest_xray"
    _make_dataset(root)
    return build_manifest(root)


def test_manifest_covers_every_image(manifest: pd.DataFrame) -> None:
    """Every written image appears exactly once with a parsed group."""
    assert len(manifest) == 20 + 40 + 2 + 10 + 10
    assert manifest["group_id"].notna().all()
    assert set(manifest["split_original"]) == {"train", "val", "test"}


def test_protocol_b_never_splits_a_patient(manifest: pd.DataFrame) -> None:
    """The grouped protocol must keep each patient wholly in one split."""
    result = make_splits(manifest, protocol="b_patient_grouped", seed=0)
    assert count_leaked_groups(result) == 0


def test_both_protocols_assign_every_image(manifest: pd.DataFrame) -> None:
    """No image may be dropped or left unassigned by either protocol."""
    for protocol in ("a_paper_compatible", "b_patient_grouped"):
        result = make_splits(manifest, protocol=protocol, seed=0)
        assert result["split"].notna().all()
        assert len(result) == len(manifest)


def test_original_test_split_is_preserved(manifest: pd.DataFrame) -> None:
    """Both protocols leave the published test split untouched.

    Results stay comparable across protocols only if the holdout is fixed.
    """
    expected = set(manifest[manifest["split_original"] == "test"]["path"])
    for protocol in ("a_paper_compatible", "b_patient_grouped"):
        result = make_splits(manifest, protocol=protocol, seed=0)
        assert set(result[result["split"] == "test"]["path"]) == expected


def test_splits_are_deterministic_given_a_seed(manifest: pd.DataFrame) -> None:
    """The same seed must reproduce the same split, or runs are not comparable."""
    for protocol in ("a_paper_compatible", "b_patient_grouped"):
        first = make_splits(manifest, protocol=protocol, seed=7)
        second = make_splits(manifest, protocol=protocol, seed=7)
        assert first["split"].equals(second["split"])


def test_unknown_protocol_rejected(manifest: pd.DataFrame) -> None:
    """An unrecognised protocol name should fail loudly."""
    with pytest.raises(ValueError, match="Unknown protocol"):
        make_splits(manifest, protocol="c_random", seed=0)


def test_split_summary_and_rows_agree(manifest: pd.DataFrame) -> None:
    """The reported summary totals must match the rows actually served."""
    result = make_splits(manifest, protocol="b_patient_grouped", seed=0)
    summary = split_summary(result)
    for split in ("train", "val", "test"):
        assert len(manifest_rows(result, split)) == summary.loc[split, "total"]
