"""Class-conditional score shift measured with one scoring function per side.

The earlier comparison put single-model out-of-fold scores against ensemble
benchmark scores. The asymmetry it found is unlikely to be an artefact of that,
but the two sides were produced by different scoring functions, so the measured
distances mix a genuine domain shift with a change in how the score was formed.

Here fold k's model is compared against itself: its own held-out validation
predictions versus its own benchmark predictions. Five paired measurements per
configuration, no ensembling anywhere.

This does not reopen the ensemble-mismatch hypothesis, which the per-fold
specificity comparison already rejected. It only harmonises the measurement.

    python3 scripts/fold_matched_score_shift.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("notebooks/results_v4")
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
#: Clip before taking log-odds. Reported at several values because a single
#: saturated score can otherwise dominate a Wasserstein distance.
CLIPS = (1e-4, 1e-6, 1e-8)


def to_group(frame, prob_column="p_pneumonia"):
    """Collapse image scores to one score per filename-derived group.

    Args:
        frame: Predictions with group_id, class_id and a probability column.
        prob_column: Name of the probability column.

    Returns:
        Tuple of (labels, probabilities) ordered by group id.
    """
    rolled = (frame.groupby("group_id")
              .agg(y=("class_id", "first"), p=(prob_column, "mean"))
              .sort_index())
    return rolled["y"].to_numpy(), rolled["p"].to_numpy()


def logit(p, clip):
    """Log-odds at an explicit clip.

    Args:
        p: Probabilities.
        clip: Distance from 0 and 1 to clamp to.

    Returns:
        Log-odds.
    """
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def main():
    bench_all = pd.read_csv(RESULTS / "predictions_benchmark_per_fold.csv")
    rows = []

    for name in CONFIGS:
        for fold in range(5):
            source = pd.read_csv(
                RESULTS / f"validation_predictions_{name}_fold{fold}.csv",
                usecols=["group_id", "class_id", "p_pneumonia"])
            target = bench_all[(bench_all["experiment"] == name)
                               & (bench_all["fold"] == fold)]
            y_s, p_s = to_group(source)
            y_t, p_t = to_group(target)

            for clip in CLIPS:
                z_s, z_t = logit(p_s, clip), logit(p_t, clip)
                entry = {"experiment": name, "fold": fold, "clip": clip,
                         "n_source": len(y_s), "n_target": len(y_t)}
                for label, cls in ((0, "normal"), (1, "pneumonia")):
                    a, b = z_s[y_s == label], z_t[y_t == label]
                    statistic, pvalue = ks_2samp(a, b)
                    entry.update({
                        f"ks_{cls}": float(statistic),
                        f"ks_p_{cls}": float(pvalue),
                        f"wasserstein_{cls}": float(wasserstein_distance(a, b)),
                        f"median_gap_{cls}": float(np.median(b) - np.median(a)),
                        f"n_source_{cls}": int((y_s == label).sum()),
                        f"n_target_{cls}": int((y_t == label).sum())})
                rows.append(entry)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "results_fold_matched_score_shift.csv", index=False)

    print("SHIFT ĐO BẰNG CÙNG MỘT MÔ HÌNH HAI BÊN (5 fold, mức group)\n")
    for clip in CLIPS:
        subset = frame[frame["clip"] == clip]
        print(f"clip = {clip:g}")
        print(f"{'cấu hình':<16} {'KS NORMAL':>18} {'KS PNEU':>18} "
              f"{'W1 NORMAL':>16} {'W1 PNEU':>14}")
        print("-" * 88)
        for name in CONFIGS:
            block = subset[subset["experiment"] == name]
            def span(column):
                values = block[column]
                return f"{values.median():.3f} [{values.min():.3f},{values.max():.3f}]"
            print(f"{LABEL[name]:<16} {span('ks_normal'):>18} "
                  f"{span('ks_pneumonia'):>18} "
                  f"{block['wasserstein_normal'].median():>10.2f} "
                  f"{block['wasserstein_pneumonia'].median():>14.2f}")
        print()

    reference = frame[frame["clip"] == 1e-6]
    print("TỈ SỐ shift NORMAL / shift PNEUMONIA (clip 1e-6, trung vị 5 fold)\n")
    for name in CONFIGS:
        block = reference[reference["experiment"] == name]
        ks_ratio = block["ks_normal"].median() / block["ks_pneumonia"].median()
        w_ratio = (block["wasserstein_normal"].median()
                   / max(block["wasserstein_pneumonia"].median(), 1e-9))
        print(f"  {LABEL[name]:<16} KS ×{ks_ratio:>5.1f}   W1 ×{w_ratio:>6.1f}")

    print("\n→ results_fold_matched_score_shift.csv")


if __name__ == "__main__":
    main()
