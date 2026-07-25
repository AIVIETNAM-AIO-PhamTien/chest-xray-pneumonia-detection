"""Reproducible data audit for the Kermany chest X-ray dataset.

Prints every number the team's handbook cites about this dataset, so a
reviewer can re-derive them instead of trusting a table. Run:

    python -m scripts.audit_dataset --root-dir ../chest_xray

Sections:

1. Split and class counts.
2. Nested-duplicate detection (the published copy contains a second
   chest_xray/chest_xray/ tree holding every image again).
3. Content-hash duplicates, within and across splits.
4. Image mode and resolution spread.
5. Patient-group audit, including the naive-vs-corrected group key that
   decides whether the original splits leak patients between them.
"""

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image

from src.dataset import find_data_root
from src.splits import build_manifest, parse_group_id


def _rule(title: str) -> None:
    """Print a section header.

    Args:
        title: Section title.
    """
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def audit_counts(manifest) -> None:
    """Print per-split class counts and the class ratio.

    Args:
        manifest: Manifest from src.splits.build_manifest.
    """
    _rule("1. Split and class counts")
    print(f"{'split':<8}{'NORMAL':>9}{'PNEUMONIA':>12}{'total':>9}{'P/N':>7}")
    for split in ("train", "val", "test"):
        subset = manifest[manifest["split_original"] == split]
        counts = subset["class_name"].value_counts()
        normal = int(counts.get("NORMAL", 0))
        pneumonia = int(counts.get("PNEUMONIA", 0))
        ratio = pneumonia / normal if normal else float("nan")
        print(
            f"{split:<8}{normal:>9,}{pneumonia:>12,}"
            f"{normal + pneumonia:>9,}{ratio:>7.2f}"
        )
    print(f"{'TOTAL':<8}{'':>9}{'':>12}{len(manifest):>9,}")


def audit_nested_tree(root: Path) -> None:
    """Report whether a nested duplicate dataset tree exists under root.

    Args:
        root: Dataset root returned by find_data_root.
    """
    _rule("2. Nested duplicate tree")
    nested = root / "chest_xray"
    outer = len(list(root.glob("*/*/*.jpeg")))
    if nested.is_dir():
        inner = len(list(nested.glob("*/*/*.jpeg")))
        print(f"FOUND nested tree: {nested}")
        print(f"  images in outer tree : {outer:,}")
        print(f"  images in nested tree: {inner:,}")
        print(f"  a recursive loader would see {outer + inner:,} instead of {outer:,}")
    else:
        print(f"No nested tree under {root} — recursive scanning is safe here.")


def audit_hashes(manifest) -> None:
    """Report exact-content duplicates within and across splits.

    Args:
        manifest: Manifest from src.splits.build_manifest.
    """
    _rule("3. Content-hash duplicates")
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for path, split in zip(manifest["path"], manifest["split_original"]):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        by_hash[digest].append(split)

    groups = [splits for splits in by_hash.values() if len(splits) > 1]
    cross = [splits for splits in groups if len(set(splits)) > 1]
    print(f"total files            : {len(manifest):,}")
    print(f"distinct content hashes: {len(by_hash):,}")
    print(f"duplicate groups       : {len(groups)}")
    print(f"  of which cross-split : {len(cross)}")
    if cross:
        print("  WARNING: identical images span splits -> evaluation is contaminated")


def audit_images(manifest) -> None:
    """Report image colour modes and resolution spread.

    Args:
        manifest: Manifest from src.splits.build_manifest.
    """
    _rule("4. Image modes and resolutions")
    modes: Counter = Counter()
    sizes: Counter = Counter()
    unreadable = 0
    for path in manifest["path"]:
        try:
            with Image.open(path) as image:
                modes[image.mode] += 1
                sizes[image.size] += 1
        except OSError:
            unreadable += 1

    print(f"unreadable files    : {unreadable}")
    for mode, count in modes.most_common():
        print(f"  mode {mode:<4}        : {count:,}")
    print(f"distinct (w, h)     : {len(sizes):,}")
    print(
        f"most common size    : {sizes.most_common(1)[0][0]} "
        f"({sizes.most_common(1)[0][1]:,} images)"
    )


def audit_patients(manifest) -> None:
    """Compare naive and corrected patient keys for cross-split overlap.

    The naive key treats ``person<N>`` as a patient. The corrected key adds
    the pneumonia subtype, because ``bacteria`` and ``virus`` each run an
    independent counter from 1. Which key is right determines whether the
    published test split can be used as a clean holdout.

    Args:
        manifest: Manifest from src.splits.build_manifest.
    """
    _rule("5. Patient groups")

    naive_re = re.compile(r"^(person\d+)_", re.IGNORECASE)
    naive: Dict[str, set] = defaultdict(set)
    corrected: Dict[str, set] = defaultdict(set)
    for filename, split in zip(manifest["filename"], manifest["split_original"]):
        match = naive_re.match(filename)
        naive[match.group(1).lower() if match else filename].add(split)
        corrected[parse_group_id(filename)].add(split)

    naive_span = sum(1 for splits in naive.values() if len(splits) > 1)
    corrected_span = sum(1 for splits in corrected.values() if len(splits) > 1)

    print(
        f"naive key      person<N>          : {len(naive):,} groups, "
        f"{naive_span} span >1 split"
    )
    print(
        f"corrected key  (subtype, person<N>): {len(corrected):,} groups, "
        f"{corrected_span} span >1 split"
    )

    subtype_ids: Dict[str, set] = defaultdict(set)
    for filename in manifest["filename"]:
        match = re.match(r"^person(\d+)_(bacteria|virus)_", filename, re.IGNORECASE)
        if match:
            subtype_ids[match.group(2).lower()].add(int(match.group(1)))

    print("\nevidence that the person counter is per-subtype, not global:")
    for subtype, ids in sorted(subtype_ids.items()):
        print(
            f"  {subtype:<9}: {len(ids):,} ids, range 1..{max(ids)}, "
            f"density {len(ids) / max(ids):.3f}"
        )
    shared = subtype_ids.get("bacteria", set()) & subtype_ids.get("virus", set())
    print(f"  numbers used by BOTH subtypes: {len(shared):,}")
    print("  (a shared counter would make this ~0; it does not)")

    per_group = Counter()
    for filename in manifest["filename"]:
        per_group[parse_group_id(filename)] += 1
    multi = sum(1 for n in per_group.values() if n > 1)
    print(
        f"\ngroups with >1 image: {multi:,} / {len(per_group):,} "
        f"(max {max(per_group.values())} images in one group)"
    )
    print("-> a validation split carved image-wise WILL straddle patients;")
    print("   use protocol b_patient_grouped for the honest number.")

    if corrected_span == 0:
        print("\nVERDICT: original train/val/test are patient-disjoint.")
        print("         The published test split is usable as a clean holdout.")
    else:
        print(
            f"\nVERDICT: {corrected_span} patient groups span splits -> "
            "build a grouped holdout instead."
        )


def main() -> None:
    """Parse arguments and run every audit section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data/raw",
        help="Directory to search for the dataset root.",
    )
    args = parser.parse_args()

    root = find_data_root(args.root_dir)
    print(f"dataset root: {root}")
    manifest = build_manifest(root)

    audit_counts(manifest)
    audit_nested_tree(root)
    audit_hashes(manifest)
    audit_images(manifest)
    audit_patients(manifest)


if __name__ == "__main__":
    main()
