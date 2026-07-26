"""Extract acquisition descriptors for every image in the frozen manifest.

Paths in the manifest point at the Kaggle mount, so they are remapped onto
whatever local copy of the dataset exists. The manifest itself is never
modified; only the features are written.

    python3 scripts/extract_nuisance_features.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nuisance_features import extract  # noqa: E402

RESULTS = Path("notebooks/results_v4")
CANDIDATE_ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]


def local_path(kaggle_path, root):
    """Map a manifest path onto the local dataset copy.

    Args:
        kaggle_path: Path recorded when the notebook ran on Kaggle.
        root: Local directory containing train/val/test.

    Returns:
        The corresponding local path.
    """
    return root / kaggle_path.split("chest_xray/")[-1]


def main():
    root = next((p for p in CANDIDATE_ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {CANDIDATE_ROOTS}")

    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    paths = [str(local_path(p, root)) for p in manifest["path"]]
    missing = [p for p in paths[:50] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Đường dẫn không tồn tại, ví dụ: {missing[0]}")

    print(f"Trích {len(paths):,} ảnh từ {root} ...", flush=True)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(extract, paths, chunksize=64))

    features = pd.DataFrame(rows).drop(columns="path")
    features.insert(0, "group_id", manifest["group_id"].to_numpy())
    features.insert(0, "class_id", manifest["class_id"].to_numpy())
    features.insert(0, "split_original", manifest["split_original"].to_numpy())
    features.insert(0, "filename", manifest["filename"].to_numpy())
    features.to_csv(RESULTS / "nuisance_feature_manifest.csv", index=False)
    print(f"→ nuisance_feature_manifest.csv  "
          f"({len(features):,} hàng × {features.shape[1]} cột)")


if __name__ == "__main__":
    main()
