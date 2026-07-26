"""Paired tests giữa các ô của thiết kế 2x2 trên cùng một tập group.

Bảng factorial trong notebook báo chênh lệch mô tả: stretch có độ đặc hiệu cao
hơn letterbox bao nhiêu điểm. Nó không nói chênh lệch đó có phân biệt được với
nhiễu lấy mẫu hay không. Bốn cấu hình chấm trên **cùng 428 group**, nên so sánh
ghép cặp mạnh hơn nhiều so với so hai tỉ lệ độc lập.

McNemar chỉ nhìn các group mà hai mô hình bất đồng, đúng trọng tâm câu hỏi.
Bootstrap ghép cặp cho khoảng tin cậy của chênh lệch độ đặc hiệu và của ΔAUC.

Cách dùng:
    python3 scripts/paired_factorial_tests.py           # trong notebooks/results_v4
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

NAMES = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LABEL = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
         "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}

D = {n: pd.read_csv(f"predictions_known_benchmark_{n}_groups.csv")
         .set_index("group_id").sort_index() for n in NAMES}
base = D["resnet18"]
for n, d in D.items():
    assert d.index.equals(base.index) and (d["label"] == base["label"]).all()
y = base["label"].to_numpy()
normal = y == 0
print(f"{len(y)} group ghép cặp; {normal.sum()} NORMAL, {(~normal).sum()} PNEUMONIA\n")

def mcnemar(a, b):
    """Chỉ tính trên NORMAL: ai báo nhầm mà người kia không."""
    pa, pb = D[a]["pred"].to_numpy()[normal], D[b]["pred"].to_numpy()[normal]
    b01 = int(((pa == 1) & (pb == 0)).sum())   # a sai, b đúng
    b10 = int(((pa == 0) & (pb == 1)).sum())   # a đúng, b sai
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return b01, b10, p

def boot_spec(a, b, reps=10000, seed=42):
    pa, pb = D[a]["pred"].to_numpy()[normal], D[b]["pred"].to_numpy()[normal]
    sa, sb = (pa == 0).astype(float), (pb == 0).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (reps, len(sa)))
    diff = sb[idx].mean(1) - sa[idx].mean(1)
    return np.percentile(diff, [2.5, 97.5])

def boot_auc(a, b, reps=10000, seed=42):
    qa, qb = D[a]["p_pneumonia"].to_numpy(), D[b]["p_pneumonia"].to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], qb[i]) - roc_auc_score(y[i], qa[i]))
    return np.percentile(out, [2.5, 97.5])

PAIRS = [("resnet18", "stretch_nhe",  "cùng augment nhẹ:  letterbox → stretch"),
         ("augment_manh", "stretch_manh", "cùng augment mạnh: letterbox → stretch"),
         ("resnet18", "augment_manh", "cùng letterbox:    nhẹ → mạnh"),
         ("stretch_nhe", "stretch_manh", "cùng stretch:      nhẹ → mạnh"),
         ("resnet18", "stretch_manh", "baseline → cấu hình bền nhất")]

print(f"{'so sánh':<40} {'ΔFP':>5} {'Δđặc hiệu':>11} {'KTC 95%':>18} {'McNemar p':>10}")
print("-" * 92)
rows = []
for a, b, desc in PAIRS:
    b01, b10, p = mcnemar(a, b)
    lo, hi = boot_spec(a, b)
    fp_a = int((D[a]["pred"].to_numpy()[normal] == 1).sum())
    fp_b = int((D[b]["pred"].to_numpy()[normal] == 1).sum())
    print(f"{desc:<40} {fp_b - fp_a:>+5d} {(lo+hi)/2:>+10.1%} "
          f"  [{lo:+.1%}, {hi:+.1%}] {p:>10.4g}")
    rows.append(dict(comparison=desc, baseline=a, variant=b, fp_baseline=fp_a,
                     fp_variant=fp_b, fp_delta=fp_b - fp_a,
                     discordant_baseline_only=b01, discordant_variant_only=b10,
                     spec_ci_low=lo, spec_ci_high=hi, mcnemar_p=p))

print(f"\n{'so sánh':<40} {'ΔAUC KTC 95%':>26}")
print("-" * 68)
for a, b, desc in PAIRS:
    lo, hi = boot_auc(a, b)
    flag = "" if lo > 0 else "  (chứa 0)"
    print(f"{desc:<40}   [{lo:+.4f}, {hi:+.4f}]{flag}")
    for r in rows:
        if r["baseline"] == a and r["variant"] == b:
            r["auc_ci_low"], r["auc_ci_high"] = lo, hi

pd.DataFrame(rows).to_csv("results_paired_tests.csv", index=False)
print("\n→ results_paired_tests.csv")
