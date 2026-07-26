"""Export a blinded package so clinicians can say what the hard normals are.

The probes establish that the models separate benchmark normals into an easy
and a hard group, and that the split is a property of the images rather than of
one classifier. They cannot say whether the hard ones are clean radiographs the
model mishandled, technically poor images, or examinations that carry an
abnormality the binary label does not name. That is a reading task.

Blinding has to survive the file itself. Names in this dataset encode the class
outright, and JPEG headers carry the quantization table that separates the
splits, so the export renames everything and re-saves as PNG. Reviewers see the
decoded picture at its native size and nothing else.

True positives are included as hidden controls. A reader told that every image
came from a normal folder will drift toward calling everything normal, and the
comparison of interest would shrink for a reason that has nothing to do with
the images.

    python3 scripts/export_clinical_review.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("notebooks/results_v4")
ROOTS = [Path("../chest_xray"), Path("data/raw/chest_xray")]
PACKAGE = Path("clinical_review_package")
CONFIG = "stretch_manh"
N_TRUE_NEGATIVE, N_TRUE_POSITIVE = 52, 20
SEED = 20260726


def stratified_sample(candidates, sizes, n, reference_sizes, seed):
    """Draw a sample matched on how many images each group contains.

    Group size tracks how a patient was imaged, so an unmatched control set
    would differ from the cases in a way unrelated to the question.

    Args:
        candidates: Group ids to draw from.
        sizes: Mapping of group id to image count.
        n: Number to draw.
        reference_sizes: Image counts of the case group being matched.
        seed: Random seed.

    Returns:
        The sampled group ids.
    """
    rng = np.random.default_rng(seed)
    wanted = pd.Series(reference_sizes).value_counts()
    pool = pd.Series(candidates)
    picked = []
    for size, count in wanted.items():
        matching = [g for g in pool if sizes[g] == size and g not in picked]
        take = min(count, len(matching))
        if take:
            picked += list(rng.choice(matching, take, replace=False))
    remaining = [g for g in pool if g not in picked]
    if len(picked) < n and remaining:
        picked += list(rng.choice(remaining, min(n - len(picked),
                                                 len(remaining)),
                                  replace=False))
    return picked[:n]


def main():
    root = next((p for p in ROOTS if p.is_dir()), None)
    if root is None:
        raise SystemExit(f"Không tìm thấy dataset trong {ROOTS}")

    manifest = pd.read_csv(RESULTS / "manifest_fold0.csv")
    benchmark = manifest[manifest["split_original"] == "test"]
    predictions = pd.read_csv(
        RESULTS / f"predictions_known_benchmark_{CONFIG}_groups.csv")

    sizes = benchmark.groupby("group_id").size().to_dict()
    false_positive = predictions[(predictions["label"] == 0)
                                 & (predictions["pred"] == 1)]["group_id"].tolist()
    true_negative = predictions[(predictions["label"] == 0)
                                & (predictions["pred"] == 0)]["group_id"].tolist()
    false_negative = predictions[(predictions["label"] == 1)
                                 & (predictions["pred"] == 0)]["group_id"].tolist()
    true_positive = predictions[(predictions["label"] == 1)
                                & (predictions["pred"] == 1)]["group_id"].tolist()

    rng = np.random.default_rng(SEED)
    chosen = {
        "false_positive": false_positive,
        "true_negative": stratified_sample(
            true_negative, sizes, N_TRUE_NEGATIVE,
            [sizes[g] for g in false_positive], SEED),
        "false_negative": false_negative,
        "true_positive": list(rng.choice(true_positive, N_TRUE_POSITIVE,
                                         replace=False)),
    }

    entries = [(group, cell) for cell, groups in chosen.items()
               for group in groups]
    order = rng.permutation(len(entries))
    images_root = PACKAGE / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "private").mkdir(exist_ok=True)

    blinded_rows, key_rows = [], []
    for position, index in enumerate(order, start=1):
        group, cell = entries[index]
        review_id = f"R{position:04d}"
        folder = images_root / review_id
        folder.mkdir(exist_ok=True)
        files = benchmark[benchmark["group_id"] == group].reset_index(drop=True)

        for number, row in files.iterrows():
            source = root / row["path"].split("chest_xray/")[-1]
            # Re-save as PNG: strips the quantization table that identifies
            # the split, and the filename that names the class.
            with Image.open(source) as handle:
                handle.convert("L").save(folder / f"{review_id}_{number + 1:02d}.png")

        blinded_rows.append({"review_id": review_id,
                             "n_images": len(files),
                             "folder": f"images/{review_id}"})
        key_rows.append({"review_id": review_id, "group_id": group,
                         "confusion_cell": cell,
                         "true_label": int(files["class_id"].iloc[0]),
                         "n_images": len(files)})

    blinded = pd.DataFrame(blinded_rows).sort_values("review_id")
    blinded.to_csv(PACKAGE / "package_manifest_blinded.csv", index=False)
    pd.DataFrame(key_rows).sort_values("review_id").to_csv(
        PACKAGE / "private" / "unblinding_key.csv", index=False)

    form = blinded[["review_id", "n_images"]].copy()
    for column in ("diagnostic_quality", "positioning_crop",
                   "pneumonia_compatible_opacity", "other_abnormality",
                   "overall_examination", "confidence_1_to_5",
                   "free_text_comment"):
        form[column] = ""
    for reviewer in ("A", "B"):
        form.to_csv(PACKAGE / f"review_form_reviewer_{reviewer}.csv", index=False)

    (PACKAGE / "review_codebook.md").write_text("""# Sổ mã hóa cho phần đọc phim

Mỗi thư mục trong `images/` là **một ca**. Nếu có nhiều ảnh, chúng thuộc cùng
một người và nên được đọc cùng nhau.

Điền vào `review_form_reviewer_A.csv` hoặc `_B.csv`. Mỗi người đọc độc lập,
không trao đổi cho tới khi cả hai đã nộp.

Chỉ mô tả những gì nhìn thấy trên phim. Không cần đoán vì sao một mô hình có
thể sai — phần đó được phân tích sau khi mở khóa.

## Các trường

**diagnostic_quality** — `adequate` | `limited` | `non-diagnostic`

**positioning_crop** — `acceptable` | `rotation` | `poor_inspiration` |
`crop_issue` | `other`

**pneumonia_compatible_opacity** — `absent` | `indeterminate` | `present`

**other_abnormality** — `none` | `atelectatic_change` |
`interstitial_or_peribronchial_change` | `effusion` | `other`

**overall_examination** — `clearly_normal` | `probably_normal` |
`indeterminate` | `probably_abnormal` | `clearly_abnormal`

**confidence_1_to_5** — 1 là rất không chắc, 5 là rất chắc

**free_text_comment** — tự do, tiếng Việt hoặc tiếng Anh

## Những điều cần biết

Bộ ảnh gồm nhiều loại ca khác nhau, không đồng nhất. Đừng giả định mọi ca đều
bình thường hay đều bất thường.

Thứ tự đã được xáo ngẫu nhiên. Số thứ tự không mang thông tin.

Nếu không kết luận được, chọn `indeterminate` — đó là câu trả lời hợp lệ và
hữu ích, không phải thất bại.
""", encoding="utf-8")

    print(f"Gói review: {len(blinded)} ca, "
          f"{sum(r['n_images'] for r in blinded_rows)} ảnh\n")
    print(pd.DataFrame(key_rows)["confusion_cell"].value_counts().to_string())
    print(f"\n→ {PACKAGE}/ (khóa giải mù nằm riêng trong private/)")


if __name__ == "__main__":
    main()
