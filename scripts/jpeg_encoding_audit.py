"""P0 — read the encoder's own settings instead of inferring them from size.

Bytes per megapixel was the first evidence that train/NORMAL was produced
differently, but file size answers to content entropy, sensor noise and
sharpness as much as to encoder settings. A radiograph with more grain costs
more bytes at identical quality.

JPEG stores its quantization tables in the file. Those are the encoder's
settings, not a consequence of the picture, so comparing them separates "saved
by a different pipeline" from "harder to compress". The distinction decides
whether the finding may be called a JPEG-encoding shortcut or has to stay the
broader compression-and-noise shortcut.

    python3 scripts/jpeg_encoding_audit.py
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("notebooks/results_v4")
CANDIDATE_ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
#: Annex K luminance table, the reference JPEG quality scaling is defined from.
STANDARD_LUMINANCE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=float)


def estimated_quality(table):
    """Recover the IJG quality setting a luminance table was scaled from.

    Args:
        table: Flattened 64-entry quantization table.

    Returns:
        Estimated quality in [1, 100], or NaN when the table is not a scaling
        of the standard one.
    """
    table = np.asarray(table, dtype=float)
    if table.size != 64:
        return float("nan")
    # IJG builds a table as round(standard * scale / 100), where scale is
    # 200 - 2q above quality 50 and 5000/q below it. Invert on the median
    # ratio so a few saturated entries cannot swing the estimate.
    scale = float(np.median(table / STANDARD_LUMINANCE)) * 100.0
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(np.clip(quality, 1.0, 100.0))


def audit(path):
    """Read encoder settings and container facts for one file.

    Args:
        path: Path to the image.

    Returns:
        Mapping describing how the file was encoded.
    """
    location = Path(path)
    with Image.open(location) as handle:
        tables = handle.quantization or {}
        serialised = {str(k): list(v) for k, v in tables.items()}
        digest = hashlib.sha256(
            json.dumps(serialised, sort_keys=True).encode()).hexdigest()[:12]
        luminance = tables.get(0)
        return {
            "filename": location.name,
            "format": handle.format,
            "mode": handle.mode,
            "width": handle.width,
            "height": handle.height,
            "progressive": bool(handle.info.get("progressive")
                                or handle.info.get("progression")),
            "n_qtables": len(tables),
            "qtable_hash": digest,
            "estimated_quality": (estimated_quality(luminance)
                                  if luminance is not None else float("nan")),
            "file_size": location.stat().st_size,
            "megapixels": handle.width * handle.height / 1e6,
        }


def main():
    root = next((p for p in CANDIDATE_ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {CANDIDATE_ROOTS}")

    rows = []
    for split in ("train", "val", "test"):
        for class_name in ("NORMAL", "PNEUMONIA"):
            folder = root / split / class_name
            for path in sorted(folder.glob("*.jpeg")):
                rows.append({"split": split, "class": class_name,
                             **audit(path)})
    frame = pd.DataFrame(rows)
    frame["bytes_per_megapixel"] = frame["file_size"] / 1e3 / frame["megapixels"]
    frame.to_csv(RESULTS / "jpeg_encoding_audit.csv", index=False)

    print(f"{len(frame):,} file\n")
    print("BẢNG LƯỢNG TỬ HÓA — thiết lập của bộ mã hóa, không phụ thuộc nội dung\n")
    print(f"{'thư mục':<20} {'n':>5} {'số qtable khác nhau':>20} "
          f"{'qtable phổ biến nhất':>22} {'tỉ lệ':>7}")
    print("-" * 80)
    for (split, class_name), block in frame.groupby(["split", "class"]):
        counts = Counter(block["qtable_hash"])
        top, top_count = counts.most_common(1)[0]
        print(f"{split + '/' + class_name:<20} {len(block):>5} "
              f"{len(counts):>20} {top:>22} {top_count / len(block):>6.1%}")

    print("\n\nCHẤT LƯỢNG JPEG ƯỚC LƯỢNG từ bảng lượng tử hóa\n")
    print(f"{'thư mục':<20} {'trung vị':>9} {'p25':>7} {'p75':>7} "
          f"{'progressive':>12} {'KB/MP trung vị':>15}")
    print("-" * 76)
    for (split, class_name), block in frame.groupby(["split", "class"]):
        quality = block["estimated_quality"].dropna()
        print(f"{split + '/' + class_name:<20} {quality.median():>9.1f} "
              f"{quality.quantile(.25):>7.1f} {quality.quantile(.75):>7.1f} "
              f"{block['progressive'].mean():>11.1%} "
              f"{block['bytes_per_megapixel'].median():>15.1f}")

    print("\n\nBẢNG LƯỢNG TỬ HÓA CÓ ĐI THEO LỚP KHÔNG?\n")
    development = frame[frame["split"].isin(["train", "val"])]
    benchmark = frame[frame["split"] == "test"]
    for name, block in (("development", development), ("benchmark", benchmark)):
        shared = (set(block[block["class"] == "NORMAL"]["qtable_hash"])
                  & set(block[block["class"] == "PNEUMONIA"]["qtable_hash"]))
        normal_only = set(block[block["class"] == "NORMAL"]["qtable_hash"]) - shared
        pneu_only = set(block[block["class"] == "PNEUMONIA"]["qtable_hash"]) - shared
        overlap_rows = block["qtable_hash"].isin(shared).mean()
        print(f"  {name:<12} qtable chỉ ở NORMAL: {len(normal_only):>4}   "
              f"chỉ ở PNEUMONIA: {len(pneu_only):>4}   "
              f"dùng chung: {len(shared):>4}   "
              f"tỉ lệ ảnh dùng qtable chung: {overlap_rows:.1%}")

    print("\n→ jpeg_encoding_audit.csv")


if __name__ == "__main__":
    main()
