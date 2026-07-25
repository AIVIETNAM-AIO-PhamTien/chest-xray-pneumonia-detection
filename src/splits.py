"""Validation-split protocols for the Kermany chest X-ray dataset.

The published dataset ships a 16-image validation split, which is far too
small to select a checkpoint on. Both protocols here therefore carve a real
validation set out of the original train+val pool and leave the original
test split untouched as the held-out set.

They differ only in how that carve is done:

- ``a_paper_compatible``: stratified split at the *image* level. This is what
  most published work and the reference notebooks do, so numbers produced
  under it are comparable with theirs. It can place two images of the same
  patient on opposite sides of the train/val boundary.
- ``b_patient_grouped``: stratified split at the *patient group* level, so
  every image of a patient lands on the same side. Validation metrics under
  this protocol are the honest ones.

Running both and reporting the gap is the point: the gap measures how much
the image-level protocol inflates validation numbers.

On patient identity
-------------------
Filenames encode the only patient handle available:

- ``person<N>_bacteria_<M>.jpeg`` / ``person<N>_virus_<M>.jpeg``
- ``IM-<N>-<M>.jpeg`` / ``NORMAL<K>-IM-<N>-<M>.jpeg``

The ``person`` counter is **not** global: ``bacteria`` and ``virus`` each run
their own near-dense sequence from 1 (1..1954 and 1..1685 respectively, with
979 numbers used by both). ``person1_bacteria_1`` and ``person1_virus_6`` are
therefore different patients, and the group key must include the subtype.

This matters. Keying on ``person<N>`` alone reports 170 patients apparently
straddling the original train and test splits; keying on
``(subtype, person<N>)`` reports zero out of 4,097 groups. The original
splits are patient-disjoint, and the 170 are an artifact of the naive key.
``scripts/audit_dataset.py`` reproduces that evidence chain.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

CLASSES = ("NORMAL", "PNEUMONIA")
PROTOCOLS = ("a_paper_compatible", "b_patient_grouped")

_PNEUMONIA_RE = re.compile(r"^person(\d+)_(bacteria|virus)_", re.IGNORECASE)
_NORMAL_RE = re.compile(r"^(?:(NORMAL\d+)-)?IM-(\d+)-", re.IGNORECASE)


def parse_group_id(filename: str) -> str:
    """Derive a patient group key from a dataset filename.

    Args:
        filename: Bare filename, e.g. "person1_bacteria_1.jpeg".

    Returns:
        A group key that is stable across the images of one patient, e.g.
        "pneumonia:bacteria:1" or "normal:normal2:1427".

    Raises:
        ValueError: If the filename matches neither known naming scheme.
            Silently bucketing unknown names together would fabricate
            patient groups, so this fails loudly instead.
    """
    match = _PNEUMONIA_RE.match(filename)
    if match:
        return f"pneumonia:{match.group(2).lower()}:{int(match.group(1))}"

    match = _NORMAL_RE.match(filename)
    if match:
        prefix = (match.group(1) or "IM").lower()
        return f"normal:{prefix}:{int(match.group(2))}"

    raise ValueError(
        f"Unrecognised filename pattern, cannot derive patient: {filename}"
    )


def build_manifest(root_dir: Union[str, Path]) -> pd.DataFrame:
    """Scan the dataset into a per-image manifest.

    Args:
        root_dir: Dataset root containing train/, val/ and test/ folders.

    Returns:
        A DataFrame with one row per image and columns: path, filename,
        split_original, class_name, class_id, group_id.

    Raises:
        FileNotFoundError: If no images are found under root_dir.
    """
    root_dir = Path(root_dir)
    rows = []
    for split in ("train", "val", "test"):
        for class_id, class_name in enumerate(CLASSES):
            class_dir = root_dir / split / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.jpeg")):
                rows.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "split_original": split,
                        "class_name": class_name,
                        "class_id": class_id,
                        "group_id": parse_group_id(path.name),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No .jpeg images found under {root_dir}")
    return pd.DataFrame(rows)


def _grouped_holdout(
    frame: pd.DataFrame, val_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train/val without splitting any patient group.

    Uses StratifiedGroupKFold so class balance is preserved as far as the
    group structure allows, then takes one fold as validation.

    Args:
        frame: Manifest rows to split.
        val_fraction: Approximate fraction to hold out for validation.
        seed: Random seed.

    Returns:
        A (train, val) tuple of DataFrames.
    """
    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(frame, frame["class_id"], groups=frame["group_id"])
    )
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()


def make_splits(
    manifest: pd.DataFrame,
    protocol: str = "b_patient_grouped",
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every image to a train/val/test split under the given protocol.

    The original test split is always preserved verbatim, so results stay
    comparable across protocols and with published numbers. Only the
    train/val boundary changes.

    Args:
        manifest: Manifest from build_manifest.
        protocol: Either "a_paper_compatible" (image-level split) or
            "b_patient_grouped" (group-aware split).
        val_fraction: Fraction of the train+val pool held out for validation.
        seed: Random seed controlling the split.

    Returns:
        A copy of the manifest with an added "split" column.

    Raises:
        ValueError: If an unknown protocol name is given.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol: {protocol!r}. Available: {list(PROTOCOLS)}"
        )

    manifest = manifest.copy()
    pool = manifest[manifest["split_original"].isin(["train", "val"])]
    test = manifest[manifest["split_original"] == "test"]

    if protocol == "a_paper_compatible":
        train, val = train_test_split(
            pool,
            test_size=val_fraction,
            stratify=pool["class_id"],
            random_state=seed,
        )
    else:
        train, val = _grouped_holdout(pool, val_fraction, seed)

    manifest["split"] = pd.NA
    manifest.loc[train.index, "split"] = "train"
    manifest.loc[val.index, "split"] = "val"
    manifest.loc[test.index, "split"] = "test"
    return manifest


def count_leaked_groups(manifest: pd.DataFrame) -> int:
    """Count patient groups that appear in more than one split.

    Args:
        manifest: Manifest carrying a "split" column, from make_splits.

    Returns:
        The number of group_ids spanning two or more splits. Zero is the
        only acceptable value for protocol B; protocol A is expected to be
        non-zero and that difference is the finding, not a bug.
    """
    per_group = manifest.groupby("group_id")["split"].nunique()
    return int((per_group > 1).sum())


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a per-split class-count table for logging and reports.

    Args:
        manifest: Manifest carrying a "split" column, from make_splits.

    Returns:
        A DataFrame indexed by split with columns per class plus "total",
        "groups" and "pneumonia_ratio".
    """
    rows = []
    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        counts = subset["class_name"].value_counts()
        normal = int(counts.get("NORMAL", 0))
        pneumonia = int(counts.get("PNEUMONIA", 0))
        rows.append(
            {
                "split": split,
                "NORMAL": normal,
                "PNEUMONIA": pneumonia,
                "total": normal + pneumonia,
                "groups": subset["group_id"].nunique(),
                "pneumonia_ratio": round(pneumonia / max(normal, 1), 2),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def manifest_rows(manifest: pd.DataFrame, split: str) -> List[Tuple[str, int]]:
    """Extract (path, label) pairs for one split.

    Args:
        manifest: Manifest carrying a "split" column, from make_splits.
        split: Split name ("train", "val" or "test").

    Returns:
        A list of (image path, class id) tuples.
    """
    subset = manifest[manifest["split"] == split]
    return list(zip(subset["path"].tolist(), subset["class_id"].tolist()))


def save_manifest(manifest: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write the manifest to CSV so a run's exact split can be re-checked.

    Args:
        manifest: Manifest to write.
        path: Destination CSV path; parent directories are created.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def load_manifest(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a previously saved manifest, if it exists.

    Args:
        path: CSV path written by save_manifest.

    Returns:
        The manifest DataFrame, or None if the file does not exist.
    """
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


__all__ = [
    "CLASSES",
    "PROTOCOLS",
    "build_manifest",
    "count_leaked_groups",
    "load_manifest",
    "make_splits",
    "manifest_rows",
    "parse_group_id",
    "save_manifest",
    "split_summary",
]
