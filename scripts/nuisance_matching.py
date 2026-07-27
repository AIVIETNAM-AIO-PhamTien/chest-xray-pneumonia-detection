"""Phase 2A.1 tier 3 — does balancing a nuisance close the normal-class gap?

Development normals score very differently from benchmark normals. The
screening step found acquisition descriptors that separate those two sets
almost perfectly. This asks the explanatory question: if the two sets are made
comparable on such a descriptor, does the score gap shrink?

Matching pairs development normals with benchmark normals using only the
descriptor and the domain label. Model scores, predictions and error status are
never consulted, so one matched cohort serves all four frozen configurations.

Every matched result is compared against random subsets of the same size and
domain composition. Without that, a distance shrinking because fewer points are
left cannot be told apart from a distance shrinking because the confound was
removed.

    python3 scripts/nuisance_matching.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nuisance_features import ALL_FEATURES, FAMILIES  # noqa: E402

RESULTS = Path("notebooks/results_v4")
CONFIGS = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
CALIPER_SD = 0.2
RANDOM_CONTROLS = 5000
BALANCE_LIMIT = 0.1
SEED = 42
CLIP = 1e-6


def logit(p):
    """Log-odds at a fixed clip.

    Args:
        p: Probabilities.

    Returns:
        Log-odds.
    """
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1.0 - CLIP)
    return np.log(p / (1.0 - p))


def smd(a, b):
    """Standardized mean difference.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Difference in means over pooled standard deviation.
    """
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 1e-12 else 0.0


def match(source_values, target_values, caliper):
    """Optimal one-to-one pairing within a caliper.

    Args:
        source_values: Descriptor values for development normals.
        target_values: Descriptor values for benchmark normals.
        caliper: Largest acceptable absolute difference within a pair.

    Returns:
        Tuple of index arrays into source and target.
    """
    cost = np.abs(target_values[:, None] - source_values[None, :])
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= caliper
    return cols[keep], rows[keep]


def distances(source_scores, target_scores):
    """Every measure of how far apart two score samples are.

    Args:
        source_scores: Log-odds from development.
        target_scores: Log-odds from the benchmark.

    Returns:
        Mapping of distance name to value.
    """
    return {
        "wasserstein": float(wasserstein_distance(source_scores, target_scores)),
        "ks": float(ks_2samp(source_scores, target_scores).statistic),
        "mean_gap": float(np.mean(target_scores) - np.mean(source_scores)),
        "median_gap": float(np.median(target_scores) - np.median(source_scores)),
    }


def random_controls(source_scores, target_scores, n_source, n_target,
                    reps=RANDOM_CONTROLS, seed=SEED):
    """Distances from random subsets of the same size, as the null.

    Args:
        source_scores: All development log-odds.
        target_scores: All benchmark log-odds.
        n_source: Size of the matched development side.
        n_target: Size of the matched benchmark side.
        reps: Number of random subsets.
        seed: Random seed.

    Returns:
        Mapping of distance name to the array of null values.
    """
    rng = np.random.default_rng(seed)
    gathered = {"wasserstein": [], "ks": [], "mean_gap": [], "median_gap": []}
    for _ in range(reps):
        a = rng.choice(source_scores, n_source, replace=False)
        b = rng.choice(target_scores, n_target, replace=False)
        for key, value in distances(a, b).items():
            gathered[key].append(value)
    return {k: np.asarray(v) for k, v in gathered.items()}


def main():
    features = pd.read_csv(RESULTS / "nuisance_feature_manifest.csv")
    grouped = (features.groupby(["group_id", "class_id", "split_original"],
                                as_index=False)[ALL_FEATURES].median())
    normals = grouped[grouped["class_id"] == 0].copy()
    normals["is_benchmark"] = (normals["split_original"] == "test").astype(int)

    screen = pd.read_csv(RESULTS / "results_normal_domain_shift.csv")
    candidates = screen[screen["passes_gate"]]["feature"].tolist()
    # One representative per family plus everything that cleared the gate.
    for family, members in FAMILIES.items():
        block = screen[screen["feature"].isin(members)]
        if len(block) and block.iloc[0]["feature"] not in candidates:
            candidates.append(block.iloc[0]["feature"])

    source = normals[normals["is_benchmark"] == 0].reset_index(drop=True)
    target = normals[normals["is_benchmark"] == 1].reset_index(drop=True)
    print(f"NORMAL: development {len(source)}, benchmark {len(target)}\n")

    # Multivariable propensity on the pre-registered feature set.
    scaler = StandardScaler().fit(normals[ALL_FEATURES])
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
    model.fit(scaler.transform(normals[ALL_FEATURES]), normals["is_benchmark"])
    normals["propensity_logit"] = model.decision_function(
        scaler.transform(normals[ALL_FEATURES]))
    source["propensity_logit"] = normals.loc[
        normals["is_benchmark"] == 0, "propensity_logit"].to_numpy()
    target["propensity_logit"] = normals.loc[
        normals["is_benchmark"] == 1, "propensity_logit"].to_numpy()

    scores = {}
    for name in CONFIGS:
        bench = pd.read_csv(
            RESULTS / f"predictions_known_benchmark_{name}_groups.csv")
        oof = pd.concat(
            [pd.read_csv(RESULTS / f"validation_predictions_{name}_fold{k}.csv",
                         usecols=["group_id", "class_id", "p_pneumonia"])
             for k in range(5)], ignore_index=True)
        oof_group = (oof.groupby("group_id")
                     .agg(y=("class_id", "first"), p=("p_pneumonia", "mean")))
        scores[name] = {
            "source": oof_group.loc[source["group_id"], "p"].to_numpy(),
            "target": bench.set_index("group_id").loc[
                target["group_id"], "p_pneumonia"].to_numpy()}

    rows = []
    for feature in candidates + ["propensity_logit"]:
        a = source[feature].to_numpy(dtype=float)
        b = target[feature].to_numpy(dtype=float)
        caliper = CALIPER_SD * np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        source_index, target_index = match(a, b, caliper)
        n_pairs = len(source_index)
        balance = (smd(a[source_index], b[target_index]) if n_pairs >= 10
                   else np.nan)
        feasible = n_pairs >= 30 and abs(balance) < BALANCE_LIMIT

        entry = {"feature": feature, "n_pairs": int(n_pairs),
                 "smd_before": smd(a, b), "smd_after": balance,
                 "balanced": bool(np.isfinite(balance)
                                  and abs(balance) < BALANCE_LIMIT),
                 "feasible": bool(feasible),
                 "match_rate": float(n_pairs / min(len(a), len(b)))}

        for name in CONFIGS:
            src = logit(scores[name]["source"])
            tgt = logit(scores[name]["target"])
            before = distances(src, tgt)
            if feasible:
                after = distances(src[source_index], tgt[target_index])
                null = random_controls(src, tgt, n_pairs, n_pairs)
                for key in ("wasserstein", "ks"):
                    attenuation = 1.0 - after[key] / max(before[key], 1e-12)
                    null_attenuation = 1.0 - null[key] / max(before[key], 1e-12)
                    entry[f"{name}_{key}_before"] = before[key]
                    entry[f"{name}_{key}_after"] = after[key]
                    entry[f"{name}_{key}_attenuation"] = attenuation
                    entry[f"{name}_{key}_null_attenuation"] = float(
                        null_attenuation.mean())
                    entry[f"{name}_{key}_p"] = float(
                        (null[key] <= after[key]).mean())
            else:
                for key in ("wasserstein", "ks"):
                    entry[f"{name}_{key}_before"] = before[key]
                    entry[f"{name}_{key}_attenuation"] = np.nan
                    entry[f"{name}_{key}_p"] = np.nan
        rows.append(entry)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "results_nuisance_matching.csv", index=False)

    print("KHẢ NĂNG CÂN BẰNG\n")
    print(f"{'feature':<24} {'SMD trước':>10} {'cặp ghép':>9} {'tỉ lệ':>7} "
          f"{'SMD sau':>9} {'dùng được':>10}")
    print("-" * 74)
    for _, row in frame.iterrows():
        after = (f"{row['smd_after']:>9.3f}" if np.isfinite(row["smd_after"])
                 else f"{'—':>9}")
        print(f"{row['feature']:<24} {row['smd_before']:>10.2f} "
              f"{row['n_pairs']:>9} {row['match_rate']:>7.1%} {after} "
              f"{'có' if row['feasible'] else 'KHÔNG':>10}")

    usable = frame[frame["feasible"]]
    if not len(usable):
        print("\nKhông feature nào cân bằng được: hai miền tách quá xa để ghép cặp.")
        print("Bản thân điều đó là bằng chứng mạnh nhất về mức tách biệt.")
    else:
        print("\n\nSUY GIẢM KHOẢNG CÁCH SCORE SAU KHI CÂN BẰNG")
        print("(W₁ giữa NORMAL nguồn và NORMAL đích; p so với subset ngẫu nhiên "
              "cùng cỡ)\n")
        for name in CONFIGS:
            print(f"--- {LABEL[name]} ---")
            print(f"{'feature':<24} {'W1 trước':>9} {'W1 sau':>8} "
                  f"{'suy giảm':>9} {'ngẫu nhiên':>11} {'p':>7}")
            for _, row in usable.iterrows():
                print(f"{row['feature']:<24} "
                      f"{row[f'{name}_wasserstein_before']:>9.2f} "
                      f"{row[f'{name}_wasserstein_after']:>8.2f} "
                      f"{row[f'{name}_wasserstein_attenuation']:>8.1%} "
                      f"{row[f'{name}_wasserstein_null_attenuation']:>10.1%} "
                      f"{row[f'{name}_wasserstein_p']:>7.4f}")
            print()

    print("→ results_nuisance_matching.csv")


if __name__ == "__main__":
    main()
