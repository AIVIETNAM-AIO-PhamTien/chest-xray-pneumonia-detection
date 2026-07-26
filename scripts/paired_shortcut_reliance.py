"""So sánh mức phụ thuộc tỉ lệ khung giữa các ô, GIỮ ghép cặp.

Notebook bootstrap FP/TN riêng cho từng cấu hình, nên bốn khoảng tin cậy chồng
nhau và không kết luận được gì về chênh lệch. Nhưng bốn cấu hình chấm trên cùng
225 group NORMAL: lấy mẫu lại chính các group đó rồi tính lại delta cho mọi cấu
hình trên cùng mẫu sẽ triệt tiêu phần biến động chung.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

NAMES = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LAB = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
       "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}

ASPECT = (pd.read_csv("manifest_fold0.csv")
          .assign(g=lambda d: d["group_id"]).groupby("g")["aspect"].median()
          if "aspect" in pd.read_csv("manifest_fold0.csv", nrows=1).columns else None)
if ASPECT is None:
    m = pd.read_csv("manifest_fold0.csv")
    from PIL import Image
    raise SystemExit("manifest thiếu cột aspect: " + ", ".join(m.columns[:12]))

D = {n: pd.read_csv(f"predictions_known_benchmark_{n}_groups.csv")
         .set_index("group_id").sort_index() for n in NAMES}
base = D[NAMES[0]]
aspect = ASPECT.reindex(base.index)
assert aspect.notna().all(), "thiếu tỉ lệ khung cho một số group"
y = base["label"].to_numpy()
normal = np.where(y == 0)[0]
A = aspect.to_numpy()
P = {n: D[n]["pred"].to_numpy() for n in NAMES}

def delta(idx, pred):
    """Cliff's delta giữa aspect của FP và TN, trên tập group idx."""
    sub = idx[pred[idx] == 1], idx[pred[idx] == 0]
    fp, tn = A[sub[0]], A[sub[1]]
    if len(fp) < 3 or len(tn) < 3: return np.nan
    d = fp[:, None] - tn[None, :]
    return (d > 0).mean() - (d < 0).mean()

print(f"{len(normal)} group NORMAL dùng chung cho cả 4 cấu hình\n")
print("Cliff's delta quan sát:")
obs = {n: delta(normal, P[n]) for n in NAMES}
for n in NAMES: print(f"  {LAB[n]:<16} {obs[n]:.3f}")

rng = np.random.default_rng(42)
REPS = 5000
draws = {n: [] for n in NAMES}
for _ in range(REPS):
    idx = rng.choice(normal, len(normal), replace=True)
    for n in NAMES:
        draws[n].append(delta(idx, P[n]))
draws = {n: np.array(v) for n, v in draws.items()}

print(f"\nCHÊNH LỆCH delta, bootstrap GHÉP CẶP ({REPS} lần):")
print(f"{'so sánh':<42} {'Δdelta':>8} {'KTC 95%':>20} {'p 2 phía':>9}")
print("-" * 84)
rows = []
for a, b, desc in [("resnet18", "stretch_nhe", "cùng nhẹ:  letterbox → stretch"),
                   ("augment_manh", "stretch_manh", "cùng mạnh: letterbox → stretch"),
                   ("resnet18", "augment_manh", "cùng letterbox: nhẹ → mạnh"),
                   ("stretch_nhe", "stretch_manh", "cùng stretch:   nhẹ → mạnh"),
                   ("resnet18", "stretch_manh", "baseline → cấu hình bền nhất")]:
    diff = draws[b] - draws[a]
    diff = diff[np.isfinite(diff)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min((diff >= 0).mean(), (diff <= 0).mean())
    flag = "" if lo > 0 or hi < 0 else "  (chứa 0)"
    print(f"{desc:<42} {obs[b]-obs[a]:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{flag} {p:>8.3f}")
    rows.append(dict(comparison=desc, baseline=a, variant=b,
                     delta_baseline=obs[a], delta_variant=obs[b],
                     delta_diff=obs[b]-obs[a], ci_low=lo, ci_high=hi, p_two_sided=p))

print("\nSpearman(tỉ lệ khung, xác suất) trên toàn bộ NORMAL — ghép cặp:")
Q = {n: D[n]["p_pneumonia"].to_numpy() for n in NAMES}
rho_obs = {n: spearmanr(A[normal], Q[n][normal])[0] for n in NAMES}
rd = {n: [] for n in NAMES}
rng = np.random.default_rng(7)
for _ in range(REPS):
    idx = rng.choice(normal, len(normal), replace=True)
    if len(np.unique(A[idx])) < 3: continue
    for n in NAMES: rd[n].append(spearmanr(A[idx], Q[n][idx])[0])
rd = {n: np.array(v) for n, v in rd.items()}
for n in NAMES: print(f"  {LAB[n]:<16} ρ = {rho_obs[n]:.3f}")
print(f"\n{'so sánh':<42} {'Δρ':>8} {'KTC 95%':>20} {'p 2 phía':>9}")
print("-" * 84)
for a, b, desc in [("resnet18", "stretch_nhe", "cùng nhẹ:  letterbox → stretch"),
                   ("augment_manh", "stretch_manh", "cùng mạnh: letterbox → stretch"),
                   ("resnet18", "stretch_manh", "baseline → cấu hình bền nhất")]:
    diff = rd[b] - rd[a]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min((diff >= 0).mean(), (diff <= 0).mean())
    flag = "" if hi < 0 or lo > 0 else "  (chứa 0)"
    print(f"{desc:<42} {rho_obs[b]-rho_obs[a]:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{flag} {p:>8.3f}")
    rows.append(dict(comparison="Spearman: " + desc, baseline=a, variant=b,
                     delta_baseline=rho_obs[a], delta_variant=rho_obs[b],
                     delta_diff=rho_obs[b]-rho_obs[a], ci_low=lo, ci_high=hi,
                     p_two_sided=p))
pd.DataFrame(rows).to_csv("results_paired_shortcut_reliance.csv", index=False)
print("\n→ results_paired_shortcut_reliance.csv")
