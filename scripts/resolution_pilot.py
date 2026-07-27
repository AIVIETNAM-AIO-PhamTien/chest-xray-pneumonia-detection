"""Phase R0 — a same-image resolution pilot, with no network retrained.

Normal images are stored at roughly three times the pixel count of pneumonia
ones, in both splits, and that survives the reduction to 224x224 as texture.
This asks whether capping source resolution removes the trace, and whether the
frozen models react to its removal.

Everything is measured on the same image under several ceilings, so a change
cannot be attributed to different pictures. Manipulation checks are computed
within class, because pathology also alters texture and a class-level contrast
would confuse the two.

    python3 scripts/resolution_pilot.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.compression import resize_for_cache  # noqa: E402
from src.data.resolution import (  # noqa: E402
    BOTTLENECKS, apply_bottleneck, would_upsample)

RESULTS = Path("notebooks/results_v4")
ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
SIZE, MODE = 224, "letterbox"
PROXIES = ("noise", "laplacian_variance", "hf_energy", "edge_density")


def proxies(array):
    """Texture descriptors on the grid the model actually reads.

    Args:
        array: 224x224 uint8 cache entry.

    Returns:
        Mapping of proxy name to value.
    """
    image = array.astype(np.float64)
    kernel = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    noise = (np.abs(ndimage.convolve(image, kernel, mode="reflect")).mean()
             * np.sqrt(np.pi / 2.0) / 6.0)
    gradient = np.hypot(ndimage.sobel(image, axis=1),
                        ndimage.sobel(image, axis=0))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean())))
    axis = np.arange(SIZE) - SIZE / 2
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    counts = np.bincount(array.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return {
        "noise": float(noise),
        "laplacian_variance": float(ndimage.laplace(image).var()),
        "hf_energy": float(spectrum[radius > SIZE / 8].sum()
                           / max(spectrum.sum(), 1e-9)),
        "edge_density": float((gradient > gradient.mean()
                               + gradient.std()).mean()),
        "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
    }


def main():
    root = next((p for p in ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {ROOTS}")

    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    paths = [root / p.split("chest_xray/")[-1] for p in manifest["path"]]

    rows = []
    for position, path in enumerate(paths):
        with Image.open(path) as handle:
            source = handle.convert("L")
            width, height = source.size
            for view, ceiling in BOTTLENECKS.items():
                capped = apply_bottleneck(source, ceiling)
                cache = resize_for_cache(capped, SIZE, MODE)
                rows.append({
                    "index": position, "view": view,
                    "group_id": manifest["group_id"].iloc[position],
                    "class_id": manifest["class_id"].iloc[position],
                    "split_original": manifest["split_original"].iloc[position],
                    "source_megapixels": width * height / 1e6,
                    "view_megapixels": capped.width * capped.height / 1e6,
                    "untouched": would_upsample(width, height, ceiling),
                    **proxies(cache)})
        if (position + 1) % 1500 == 0:
            print(f"  {position + 1:,}/{len(paths):,}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "results_resolution_pilot_proxies.csv", index=False)

    print("\nẢNH ĐÃ NHỎ HƠN NGƯỠNG NÊN GIỮ NGUYÊN (không bao giờ upsample)\n")
    for view in list(BOTTLENECKS)[1:]:
        block = frame[frame["view"] == view]
        parts = []
        for class_id, name in ((0, "NORMAL"), (1, "PNEUMONIA")):
            subset = block[block["class_id"] == class_id]
            parts.append(f"{name} {subset['untouched'].mean():>5.1%}")
        print(f"  {view:<8} " + "   ".join(parts))

    print("\n\nKIỂM TRA CHÍNH — tương quan (độ phân giải gốc, proxy @224)")
    print("tính RIÊNG trong từng lớp, vì bệnh lý cũng làm đổi kết cấu\n")
    print(f"{'proxy':<20} {'lớp':<11} "
          + "".join(f"{v:>10}" for v in BOTTLENECKS))
    print("-" * 74)
    correlations = {}
    for proxy in PROXIES:
        for class_id, name in ((0, "NORMAL"), (1, "PNEUMONIA")):
            values = []
            for view in BOTTLENECKS:
                block = frame[(frame["view"] == view)
                              & (frame["class_id"] == class_id)]
                values.append(spearmanr(block["source_megapixels"],
                                        block[proxy]).statistic)
            correlations[(proxy, name)] = values
            print(f"{proxy:<20} {name:<11} "
                  + "".join(f"{v:>10.3f}" for v in values))

    print("\nMỨC GIẢM |rho| so với native — cổng đã đăng ký: >= 50%\n")
    print(f"{'proxy':<20} {'lớp':<11} "
          + "".join(f"{v:>10}" for v in list(BOTTLENECKS)[1:]))
    print("-" * 62)
    for (proxy, name), values in correlations.items():
        base = abs(values[0])
        drops = [1 - abs(v) / base if base > 1e-9 else np.nan
                 for v in values[1:]]
        print(f"{proxy:<20} {name:<11} "
              + "".join(f"{d:>9.0%} " for d in drops))

    print("\n\nTÁCH BIỆT MIỀN TRONG RIÊNG NORMAL (development vs benchmark)\n")
    print(f"{'view':<10} {'domain AUC':>11} {'KS':>8} {'W1 chuẩn hóa':>14}")
    print("-" * 46)
    domain_rows = []
    for view in BOTTLENECKS:
        block = frame[(frame["view"] == view) & (frame["class_id"] == 0)]
        development = block[block["split_original"] != "test"]["noise"].to_numpy()
        benchmark = block[block["split_original"] == "test"]["noise"].to_numpy()
        labels = np.r_[np.zeros(len(development)), np.ones(len(benchmark))]
        values = np.r_[development, benchmark]
        auc = roc_auc_score(labels, values)
        entry = {"view": view,
                 "domain_auc_normal": float(max(auc, 1 - auc)),
                 "ks": float(ks_2samp(development, benchmark).statistic),
                 "wasserstein_standardised": float(
                     wasserstein_distance(development, benchmark)
                     / values.std(ddof=1))}
        domain_rows.append(entry)
        print(f"{view:<10} {entry['domain_auc_normal']:>11.4f} "
              f"{entry['ks']:>8.3f} {entry['wasserstein_standardised']:>14.3f}")

    pd.DataFrame(domain_rows).to_csv(
        RESULTS / "results_resolution_pilot_domain.csv", index=False)
    print("\n→ results_resolution_pilot_proxies.csv")
    print("→ results_resolution_pilot_domain.csv")


if __name__ == "__main__":
    main()
