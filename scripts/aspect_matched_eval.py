"""Đo lại AUC trên tập con có tỉ lệ khung khớp giữa hai lớp.

Tỉ lệ khung một mình đã đạt AUC 0,70 trên test. Câu hỏi còn treo là mô hình có
dùng nó hay không. Cách hỏi trực tiếp: dựng một tập con mà đặc trưng đó KHÔNG
còn phân biệt được lớp, rồi đo lại mô hình trên đó.

Ghép cặp 1-1 tối ưu (Hungarian) mỗi group PNEUMONIA với một group NORMAL có tỉ
lệ khung gần nhất, loại cặp vượt caliper 0,2 SD gộp. Ghép chỉ dùng tỉ lệ khung
và nhãn, không đụng tới điểm số mô hình, nên tập con giống nhau cho cả 4 cấu
hình.

Đối chứng bắt buộc: một tập con ngẫu nhiên cùng cỡ. Nếu không có nó thì không
phân biệt được "AUC giảm vì hết confound" với "AUC giảm vì ít mẫu đi".
"""
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

NAMES = ["resnet18", "stretch_nhe", "augment_manh", "stretch_manh"]
LAB = {"resnet18": "letterbox+nhẹ", "stretch_nhe": "stretch+nhẹ",
       "augment_manh": "letterbox+mạnh", "stretch_manh": "stretch+mạnh"}
CALIPER_SD, SEED, REPS = 0.2, 42, 5000

m = pd.read_csv("manifest_fold0.csv")
aspect_by_group = m.groupby("group_id")["aspect"].median()
D = {n: pd.read_csv(f"predictions_known_benchmark_{n}_groups.csv")
         .set_index("group_id").sort_index() for n in NAMES}
base = D[NAMES[0]]
A = aspect_by_group.reindex(base.index).to_numpy()
y = base["label"].to_numpy()
assert np.isfinite(A).all()

pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
caliper = CALIPER_SD * np.sqrt((A[pos].var(ddof=1) + A[neg].var(ddof=1)) / 2)

cost = np.abs(A[pos][:, None] - A[neg][None, :])
ri, ci = linear_sum_assignment(cost)
keep = cost[ri, ci] <= caliper
mp, mn = pos[ri[keep]], neg[ci[keep]]
matched = np.concatenate([mp, mn])

def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

def directed_auc(labels, score):
    """AUC theo chiều đã khóa trên development: tỉ lệ lớn hơn -> PNEUMONIA."""
    return roc_auc_score(labels, score)

print(f"caliper = {CALIPER_SD} SD = {caliper:.4f} đơn vị tỉ lệ khung")
print(f"ghép được {keep.sum()} cặp; bỏ {(~keep).sum()} cặp vượt caliper")
print(f"tập đã khớp: {len(matched)} group ({len(mp)} PNEUMONIA + {len(mn)} NORMAL)\n")

print("KIỂM TRA: confound đã tắt chưa?")
print(f"{'':22} {'toàn bộ':>18} {'đã khớp':>18}")
print("-" * 60)
print(f"{'chênh TB (SMD)':<22} {smd(A[pos], A[neg]):>18.3f} {smd(A[mp], A[mn]):>18.3f}")
print(f"{'AUC chỉ dùng tỉ lệ':<22} {directed_auc(y, A):>18.4f} "
      f"{directed_auc(y[matched], A[matched]):>18.4f}")
print(f"{'trung vị PNEUMONIA':<22} {np.median(A[pos]):>18.3f} {np.median(A[mp]):>18.3f}")
print(f"{'trung vị NORMAL':<22} {np.median(A[neg]):>18.3f} {np.median(A[mn]):>18.3f}")

rng = np.random.default_rng(SEED)
n_pos, n_neg = len(mp), len(mn)

print(f"\n\nAUC MÔ HÌNH  (KTC bootstrap 95%, {REPS} lần)")
print(f"{'cấu hình':<16} {'toàn bộ':>9} {'đã khớp':>26} {'ngẫu nhiên cùng cỡ':>26} {'chênh':>8}")
print("-" * 92)
rows = []
for n in NAMES:
    q = D[n]["p_pneumonia"].to_numpy()
    full = directed_auc(y, q)
    match = directed_auc(y[matched], q[matched])

    mb, rb = [], []
    for _ in range(REPS):
        i = np.concatenate([rng.choice(mp, n_pos, True), rng.choice(mn, n_neg, True)])
        if len(np.unique(y[i])) == 2: mb.append(directed_auc(y[i], q[i]))
        j = np.concatenate([rng.choice(pos, n_pos, False), rng.choice(neg, n_neg, False)])
        rb.append(directed_auc(y[j], q[j]))
    mlo, mhi = np.percentile(mb, [2.5, 97.5])
    rlo, rhi = np.percentile(rb, [2.5, 97.5])
    rmean = float(np.mean(rb))
    print(f"{LAB[n]:<16} {full:>9.4f}   {match:.4f} [{mlo:.4f},{mhi:.4f}]"
          f"   {rmean:.4f} [{rlo:.4f},{rhi:.4f}] {match - rmean:>+8.4f}")
    rows.append(dict(experiment=n, auc_full=full, auc_matched=match,
                     auc_matched_ci_low=mlo, auc_matched_ci_high=mhi,
                     auc_random_subset_mean=rmean, auc_random_ci_low=rlo,
                     auc_random_ci_high=rhi, auc_matched_minus_random=match - rmean,
                     n_matched_pairs=int(keep.sum()),
                     aspect_auc_full=directed_auc(y, A),
                     aspect_auc_matched=directed_auc(y[matched], A[matched]),
                     smd_full=smd(A[pos], A[neg]), smd_matched=smd(A[mp], A[mn]),
                     caliper_sd=CALIPER_SD, caliper=float(caliper)))

pd.DataFrame(rows).to_csv("results_aspect_matched_eval.csv", index=False)
print("\n→ results_aspect_matched_eval.csv")
print()
print("CÁCH ĐỌC")
print("  AUC giữ nguyên sau khi khớp  -> xếp hạng KHÔNG dựa chủ yếu vào tỉ lệ khung.")
print("  AUC tụt dưới mức tập ngẫu nhiên -> phần xếp hạng đó đến từ tỉ lệ khung.")
print("  Tập đã khớp cố ý giàu ảnh NORMAL khung giống PNEUMONIA, nên độ đặc hiệu")
print("  trên đó thấp hơn là do chọn mẫu, không phải do bỏ confound. So nó với")
print("  tập NORMAL ngẫu nhiên cùng cỡ, đừng so với toàn bộ.")
