"""Draw every figure the report uses, from the frozen predictions.

Figures are regenerated rather than exported by hand so that a number in the
text and the same number in a chart cannot drift apart.

    python3 scripts/build_report_figures.py
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torchvision import models  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import find_data_root  # noqa: E402
from src.evaluation.selection import (  # noqa: E402
    exact_threshold_at_sensitivity,
    high_sensitivity_average_specificity,
)
from src.explainability import GradCAM, resolve_target_layer  # noqa: E402

V4, V5, V6 = (
    Path("notebooks/results_v4"),
    Path("notebooks/results_v5"),
    Path("notebooks/results_v6"),
)
#: The Overleaf project sits beside the repository, not inside it.
OUT = Path("../overleaf/Figures")
FINAL_XAI = Path("artifacts/final/explainability")
NORMAL, PNEUMONIA, ACCENT, GREY = "#2a78d6", "#eb6834", "#1a9e6f", "#8a8a8a"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)


def load():
    """Group-level predictions for the three trained models.

    Returns:
        Tuple of (out-of-fold frame, benchmark frame).
    """

    def oof(directory, name):
        path = directory / f"predictions_oof_{name}_groups.csv"
        if path.exists():
            frame = pd.read_csv(path)[["group_id", "label", "p_pneumonia"]]
        else:
            parts = [
                pd.read_csv(
                    directory / f"validation_predictions_{name}_fold{k}.csv",
                    usecols=["group_id", "class_id", "p_pneumonia"],
                )
                for k in range(5)
            ]
            frame = (
                pd.concat(parts, ignore_index=True)
                .groupby("group_id", as_index=False)
                .agg(label=("class_id", "first"), p_pneumonia=("p_pneumonia", "mean"))
            )
        return frame.set_index("group_id").sort_index()

    def bench(directory, name):
        return (
            pd.read_csv(directory / f"predictions_known_benchmark_{name}_groups.csv")
            .set_index("group_id")[["label", "p_pneumonia"]]
            .sort_index()
        )

    sources = {
        "ResNet18": (V4, "stretch_manh"),
        "DenseNet121": (V5, "densenet121_robust"),
        "DeiT-Small": (V6, "deit_small"),
    }
    o = {k: oof(*v)["p_pneumonia"] for k, v in sources.items()}
    b = {k: bench(*v)["p_pneumonia"] for k, v in sources.items()}
    o["Ensemble"] = (o["ResNet18"] + o["DenseNet121"]) / 2
    b["Ensemble"] = (b["ResNet18"] + b["DenseNet121"]) / 2
    y_o = oof(*sources["ResNet18"])["label"]
    y_b = bench(*sources["ResNet18"])["label"]
    return pd.DataFrame(o).assign(label=y_o), pd.DataFrame(b).assign(label=y_b)


def figure_acquisition(manifest, features):
    """The acquisition confound: JPEG quality and its pixel-visible trace."""
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.1))

    quality = [95.5, 75.0, 75.0, 75.0]
    colours = [PNEUMONIA if q > 80 else GREY for q in quality]
    axes[0].bar(range(4), quality, color=colours, width=0.62)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(
        ["train\nNORMAL", "train\nPNEU", "test\nNORMAL", "test\nPNEU"], fontsize=8
    )
    axes[0].set_ylabel("JPEG quality (ước lượng)")
    axes[0].set_ylim(60, 102)
    axes[0].set_title("(a) Thiết lập bộ mã hoá")
    for index, value in enumerate(quality):
        axes[0].text(index, value + 1, f"{value:.1f}", ha="center", fontsize=8)

    joined = manifest.merge(features, on="filename", how="inner")
    joined["group"] = (
        joined["split_original"].where(joined["split_original"] == "test", "train")
        + "/"
        + joined["class_name"]
    )
    order = ["train/NORMAL", "train/PNEUMONIA", "test/NORMAL", "test/PNEUMONIA"]
    data = [joined.loc[joined["group"] == g, "file_size_per_pixel"] for g in order]
    parts = axes[1].boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
    for patch, colour in zip(parts["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    for element in ("medians", "whiskers", "caps"):
        for line in parts[element]:
            line.set_color("#333")
    axes[1].set_xticklabels(
        ["train\nNORMAL", "train\nPNEU", "test\nNORMAL", "test\nPNEU"], fontsize=8
    )
    axes[1].set_ylabel("byte trên mỗi pixel")
    axes[1].set_title("(b) Hệ quả trên dung lượng")

    development = joined[joined["split_original"] != "test"]
    for label, name, colour in ((0, "NORMAL", NORMAL), (1, "PNEUMONIA", PNEUMONIA)):
        axes[2].hist(
            development.loc[development["class_id"] == label, "noise_estimate"],
            bins=np.linspace(0, 3.5, 55),
            color=colour,
            alpha=0.68,
            label=name,
        )
    axes[2].set_xlabel("ước lượng nhiễu Immerkaer")
    axes[2].set_ylabel("số ảnh")
    axes[2].set_title("(c) Dấu vết nhìn thấy trong pixel")
    axes[2].legend(fontsize=8)

    figure.savefig(OUT / "fig_acquisition_confound.pdf")
    plt.close(figure)


def locked_thresholds(oof):
    """Each model's operating point, fixed on development data.

    Deriving it from benchmark labels instead would be an oracle threshold and
    would flatter every model, which is exactly the mistake this project spent
    its audit chasing.

    Args:
        oof: Out-of-fold group predictions with a label column.

    Returns:
        Mapping of model name to threshold.
    """
    y = oof["label"].to_numpy()
    return {
        name: exact_threshold_at_sensitivity(y, oof[name].to_numpy())
        for name in oof.columns
        if name != "label"
    }


def figure_ladder(oof, bench):
    """False positives per model, and where the ceiling sits."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    y = bench["label"].to_numpy()
    thresholds = locked_thresholds(oof)
    models = ["ResNet18", "DenseNet121", "DeiT-Small", "Ensemble"]

    counts = []
    for name in models:
        predicted = bench[name].to_numpy() >= thresholds[name]
        counts.append(int((predicted & (y == 0)).sum()))

    colours = [GREY, GREY, GREY, ACCENT]
    bars = axes[0].bar(range(4), counts, color=colours, width=0.6)
    axes[0].axhline(22, color=PNEUMONIA, linestyle="--", lw=1.2)
    axes[0].set_xlim(-0.55, 4.25)
    axes[0].text(
        3.62, 20.5, "trần\noracle 22", color=PNEUMONIA, fontsize=8, va="center"
    )
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(
        ["ResNet18", "DenseNet121", "DeiT-S", "Ensemble"], fontsize=8
    )
    axes[0].set_ylabel("số ca báo nhầm (trên 225)")
    axes[0].set_title("(a) Báo nhầm ở độ nhạy ≥97%")
    axes[0].set_ylim(0, 88)
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            count + 2,
            str(count),
            ha="center",
            fontsize=9,
        )

    for name, colour, style in (
        ("ResNet18", GREY, ":"),
        ("DenseNet121", GREY, "-."),
        ("DeiT-Small", "#c9a227", "--"),
        ("Ensemble", ACCENT, "-"),
    ):
        fpr, tpr, _ = roc_curve(y, bench[name])
        axes[1].plot(fpr, tpr, color=colour, linestyle=style, lw=1.5, label=f"{name}")
    axes[1].axhline(0.97, color=PNEUMONIA, lw=1, alpha=0.6)
    axes[1].text(0.42, 0.973, "độ nhạy 97%", color=PNEUMONIA, fontsize=8)
    axes[1].set_xlim(0, 0.6)
    axes[1].set_ylim(0.9, 1.005)
    axes[1].set_xlabel("tỉ lệ dương tính giả")
    axes[1].set_ylabel("độ nhạy")
    axes[1].set_title("(b) ROC ở vùng vận hành")
    axes[1].legend(fontsize=8, loc="lower right")

    figure.savefig(OUT / "fig_model_ladder.pdf")
    plt.close(figure)


def figure_saturation(oof, bench):
    """Why internal validation could not rank the candidates."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    models = ["ResNet18", "DenseNet121", "DeiT-Small", "Ensemble"]

    oof_spec, bench_spec = [], []
    y_o, y_b = oof["label"].to_numpy(), bench["label"].to_numpy()
    for name in models:
        p_o, p_b = oof[name].to_numpy(), bench[name].to_numpy()
        threshold = exact_threshold_at_sensitivity(y_o, p_o)
        oof_spec.append(float((p_o[y_o == 0] < threshold).mean()))
        bench_spec.append(float((p_b[y_b == 0] < threshold).mean()))

    positions = np.arange(4)
    axes[0].plot(positions, oof_spec, "o-", color=NORMAL, lw=1.6, label="out-of-fold")
    axes[0].plot(
        positions, bench_spec, "s-", color=PNEUMONIA, lw=1.6, label="known benchmark"
    )
    for index in positions:
        axes[0].plot(
            [index, index],
            [bench_spec[index], oof_spec[index]],
            color=GREY,
            lw=0.8,
            zorder=0,
        )
        axes[0].text(
            index + 0.06,
            (oof_spec[index] + bench_spec[index]) / 2,
            f"{oof_spec[index] - bench_spec[index]:.2f}",
            fontsize=7.5,
            color=GREY,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        ["ResNet18", "DenseNet121", "DeiT-S", "Ensemble"], fontsize=8
    )
    axes[0].set_ylabel("độ đặc hiệu @ độ nhạy 97%")
    axes[0].set_ylim(0.6, 1.02)
    axes[0].set_title("(a) Khoảng cách OOF → benchmark")
    axes[0].legend(fontsize=8, loc="lower left")

    budgets = range(1, 7)
    oracle = []
    probs, labels = bench["Ensemble"].to_numpy(), y_b
    for budget in budgets:
        best = min(
            int(((labels == 0) & (probs >= t)).sum())
            for t in np.unique(probs)
            if int(((labels == 1) & (probs < t)).sum()) <= budget
        )
        oracle.append(best)
    axes[1].plot(list(budgets), oracle, "o-", color=ACCENT, lw=1.6)
    axes[1].axhline(40, color=GREY, linestyle="--", lw=1.1)
    axes[1].text(5.9, 41.4, "đạt được: 40", ha="right", fontsize=8, color=GREY)
    axes[1].axhline(20, color=PNEUMONIA, linestyle=":", lw=1.2)
    axes[1].text(
        5.9, 17.4, "mục tiêu 20 — ngoài tầm", ha="right", fontsize=8, color=PNEUMONIA
    )
    axes[1].set_xlabel("số ca bỏ sót cho phép")
    axes[1].set_ylabel("báo nhầm thấp nhất có thể")
    axes[1].set_ylim(14, 45)
    axes[1].set_title("(b) Trần của xếp hạng hiện tại")

    figure.savefig(OUT / "fig_saturation_and_ceiling.pdf")
    plt.close(figure)


def figure_transitions(oof, bench):
    """Which cases each step fixed, and which it broke."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    y = bench["label"].to_numpy()
    negative = y == 0
    thresholds = locked_thresholds(oof)

    def decisions(name):
        return (bench[name].to_numpy() >= thresholds[name]).astype(int)[negative]

    pairs = [
        ("ResNet18", "Ensemble"),
        ("DenseNet121", "Ensemble"),
        ("DeiT-Small", "Ensemble"),
    ]
    fixed, broken = [], []
    for a, b in pairs:
        left, right = decisions(a), decisions(b)
        fixed.append(int(((left == 1) & (right == 0)).sum()))
        broken.append(int(((left == 0) & (right == 1)).sum()))

    positions = np.arange(3)
    axes[0].barh(positions + 0.16, fixed, height=0.3, color=ACCENT, label="sửa được")
    axes[0].barh(
        positions - 0.16,
        [-value for value in broken],
        height=0.3,
        color=PNEUMONIA,
        label="phá hỏng",
    )
    axes[0].axvline(0, color="#333", lw=0.8)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([a for a, _ in pairs], fontsize=8)
    axes[0].set_xlabel("số group NORMAL")
    axes[0].set_title("(a) Ensemble so với từng mô hình")
    axes[0].legend(fontsize=8, loc="lower right")
    for index, (f, b) in enumerate(zip(fixed, broken)):
        axes[0].text(f + 1, index + 0.16, str(f), va="center", fontsize=8)
        if b:
            axes[0].text(
                -b - 1, index - 0.16, str(b), va="center", ha="right", fontsize=8
            )

    scores = bench["Ensemble"].to_numpy()
    for label, name, colour in ((0, "NORMAL", NORMAL), (1, "PNEUMONIA", PNEUMONIA)):
        axes[1].hist(
            scores[y == label],
            bins=np.linspace(0, 1, 45),
            color=colour,
            alpha=0.68,
            label=name,
        )
    axes[1].axvline(thresholds["Ensemble"], color="#333", lw=1.3)
    axes[1].text(
        thresholds["Ensemble"] - 0.03,
        axes[1].get_ylim()[1] * 0.30,
        f"ngưỡng {thresholds['Ensemble']:.3f}".replace(".", ","),
        fontsize=8,
        ha="right",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("xác suất PNEUMONIA (mức group)")
    axes[1].set_ylabel("số group (log)")
    axes[1].set_title("(b) Phân bố điểm số của mô hình cuối")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.16, 1.0))

    figure.savefig(OUT / "fig_transitions.pdf")
    plt.close(figure)


def figure_hsas():
    """Why a whole-case metric could not separate the candidates."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    labels = np.r_[np.zeros(400, int), np.ones(400, int)]
    base = np.r_[np.linspace(0.02, 0.52, 400), np.linspace(0.50, 0.99, 400)]
    damaged = base.copy()
    damaged[:9] = np.linspace(0.502, 0.515, 9)

    for probs, name, colour, style in (
        (base, "mô hình A", ACCENT, "-"),
        (damaged, "mô hình B", PNEUMONIA, "--"),
    ):
        fpr, tpr, _ = roc_curve(labels, probs)
        axes[0].plot(fpr, tpr, color=colour, linestyle=style, lw=1.6, label=name)
    axes[0].axhspan(0.97, 1.0, color=NORMAL, alpha=0.12)
    axes[0].text(0.085, 0.9755, "vùng HSAS@97", color=NORMAL, fontsize=8)
    axes[0].set_xlim(0, 0.14)
    axes[0].set_ylim(0.955, 1.004)
    axes[0].set_xlabel("tỉ lệ dương tính giả")
    axes[0].set_ylabel("độ nhạy")
    axes[0].set_title("(a) Hai đường cong gần như trùng nhau")
    axes[0].legend(fontsize=8, loc="lower right")

    from sklearn.metrics import roc_auc_score

    values = [
        ("AUC toàn cục", roc_auc_score(labels, base), roc_auc_score(labels, damaged)),
        (
            "HSAS@97",
            high_sensitivity_average_specificity(labels, base),
            high_sensitivity_average_specificity(labels, damaged),
        ),
    ]
    positions = np.arange(2)
    axes[1].bar(
        positions - 0.17,
        [v[1] for v in values],
        width=0.32,
        color=ACCENT,
        label="mô hình A",
    )
    axes[1].bar(
        positions + 0.17,
        [v[2] for v in values],
        width=0.32,
        color=PNEUMONIA,
        label="mô hình B",
    )
    for index, (_, a, b) in enumerate(values):
        axes[1].text(
            index, max(a, b) + 0.012, f"chênh {a - b:.4f}", ha="center", fontsize=8.5
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([v[0] for v in values])
    axes[1].set_ylim(0, 1.32)
    axes[1].set_ylabel("giá trị")
    axes[1].set_title("(b) Chỉ một chỉ số nhìn thấy khác biệt")
    axes[1].legend(fontsize=8, loc="upper center", ncol=2)

    figure.savefig(OUT / "fig_hsas.pdf")
    plt.close(figure)


def _build_cam_model(architecture, checkpoint, device):
    """Restore one final-model member without downloading pretrained weights."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_cam_inputs(cases, data_root, device):
    """Load the selected images with the exact stretch/eval preprocessing."""
    arrays, tensors = [], []
    for row in cases:
        path = data_root / "test" / row["class_name"] / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Grad-CAM source image not found: {path}")
        with Image.open(path) as image:
            gray = (
                np.asarray(
                    image.convert("L").resize((224, 224), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
        arrays.append(gray)
        tensor = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)
        tensors.append(tensor)

    batch = torch.stack(tensors).to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return arrays, (batch - mean) / std


def _image_prediction_frame():
    """Merge the two member families' saved image-level benchmark scores."""
    resnet = pd.read_csv(
        V4 / "predictions_known_benchmark_stretch_manh_images.csv",
        usecols=[
            "filename",
            "class_name",
            "class_id",
            "group_id",
            "p_pneumonia",
        ],
    ).rename(columns={"p_pneumonia": "p_resnet"})
    densenet = pd.read_csv(
        V5 / "predictions_known_benchmark_densenet121_robust_images.csv",
        usecols=["filename", "group_id", "p_pneumonia"],
    ).rename(columns={"p_pneumonia": "p_densenet"})
    images = resnet.merge(
        densenet,
        on=["filename", "group_id"],
        how="inner",
        validate="one_to_one",
    )
    images["p_ensemble_image"] = (images["p_resnet"] + images["p_densenet"]) / 2
    return images


def _select_gradcam_cases(oof, bench):
    """Select one boundary case per confusion cell by a fixed rule.

    The decision is made at group level using the locked OOF threshold.  Within
    each selected group, the image whose ensemble score is closest to the group
    mean is used.  This avoids choosing a visually convenient image.
    """
    threshold = locked_thresholds(oof)["Ensemble"]
    groups = bench[["label", "Ensemble"]].rename(
        columns={"Ensemble": "p_ensemble_group"}
    )
    groups["pred"] = (groups["p_ensemble_group"] >= threshold).astype(int)
    images = _image_prediction_frame()

    definitions = [
        ("TN", 0, 0),
        ("FP", 0, 1),
        ("FN", 1, 0),
        ("TP", 1, 1),
    ]
    selected = []
    for category, label, prediction in definitions:
        candidates = groups[
            (groups["label"] == label) & (groups["pred"] == prediction)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No {category} group available for Grad-CAM")
        candidates["distance_to_threshold"] = (
            candidates["p_ensemble_group"] - threshold
        ).abs()
        group_id = candidates["distance_to_threshold"].idxmin()
        group_row = candidates.loc[group_id]

        group_images = images[images["group_id"] == group_id].copy()
        group_images["distance_to_group_mean"] = (
            group_images["p_ensemble_image"] - group_row["p_ensemble_group"]
        ).abs()
        image_row = group_images.sort_values(
            ["distance_to_group_mean", "filename"]
        ).iloc[0]
        selected.append(
            {
                "category": category,
                "group_id": group_id,
                "filename": image_row["filename"],
                "class_name": image_row["class_name"],
                "label": int(label),
                "prediction": int(prediction),
                "p_ensemble_group": float(group_row["p_ensemble_group"]),
                "p_ensemble_image": float(image_row["p_ensemble_image"]),
                "threshold": float(threshold),
                "selection_rule": (
                    "group nearest locked threshold within confusion cell; "
                    "image nearest group mean"
                ),
            }
        )
    return selected


def _border_mass(cam, fraction=0.15):
    """Fraction of Grad-CAM activation in a geometric image border."""
    height, width = cam.shape
    vertical = max(1, round(height * fraction))
    horizontal = max(1, round(width * fraction))
    border = np.zeros_like(cam, dtype=bool)
    border[:vertical], border[-vertical:] = True, True
    border[:, :horizontal], border[:, -horizontal:] = True, True
    return float(cam[border].sum() / max(float(cam.sum()), 1e-12))


def _member_cams(architecture, checkpoints, inputs, cases, device):
    """Compute one map per fold, plus auditable member-level measurements."""
    maps, records = [], []
    for fold, checkpoint in enumerate(checkpoints):
        model = _build_cam_model(architecture, checkpoint, device)
        target_layer = resolve_target_layer(model, architecture)
        with GradCAM(model, target_layer) as explainer:
            member_maps, logits = explainer(inputs, target_class=1)
        member_maps = member_maps.cpu().numpy()
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        maps.append(member_maps)
        for case, cam, probability in zip(cases, member_maps, probabilities):
            records.append(
                {
                    "architecture": architecture,
                    "fold": fold,
                    "checkpoint": checkpoint.name,
                    "category": case["category"],
                    "group_id": case["group_id"],
                    "filename": case["filename"],
                    "p_pneumonia": float(probability),
                    "border_mass_15pct": _border_mass(cam),
                }
            )
        del model, logits, member_maps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    return np.stack(maps), records


def figure_gradcam(oof, bench):
    """Create member-consensus Grad-CAM artifacts for the final ensemble."""
    resnet_checkpoints = sorted(V4.glob("stretch_manh_fold*.pth"))
    densenet_checkpoints = sorted(V5.glob("densenet121_robust_fold*.pth"))
    if len(resnet_checkpoints) != 5 or len(densenet_checkpoints) != 5:
        print(
            "Bỏ qua fig_gradcam_final.pdf: cần đủ 5 checkpoint ResNet18 và "
            "5 checkpoint DenseNet121."
        )
        return

    data_root = find_data_root(
        [Path("../chest_xray"), Path("data/raw"), Path("../data")]
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cases = _select_gradcam_cases(oof, bench)
    originals, inputs = _load_cam_inputs(cases, data_root, device)
    resnet_maps, resnet_records = _member_cams(
        "resnet18", resnet_checkpoints, inputs, cases, device
    )
    densenet_maps, densenet_records = _member_cams(
        "densenet121", densenet_checkpoints, inputs, cases, device
    )
    medians = {
        "ResNet18 — trung vị 5 fold": np.median(resnet_maps, axis=0),
        "DenseNet121 — trung vị 5 fold": np.median(densenet_maps, axis=0),
    }

    figure, axes = plt.subplots(
        3,
        len(cases),
        figsize=(3.0 * len(cases), 8.1),
        constrained_layout=True,
    )
    heat_cmap = matplotlib.colormaps["jet"]
    for column, (case, original) in enumerate(zip(cases, originals)):
        axes[0, column].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(
            f"{case['category']} | nhãn {case['class_name']}\n"
            f"$p_g$={case['p_ensemble_group']:.3f}, "
            f"$t$={case['threshold']:.3f}",
            fontsize=9,
        )
        for row, (name, maps) in enumerate(medians.items(), start=1):
            coloured = heat_cmap(maps[column])[..., :3]
            base = np.repeat(original[..., None], 3, axis=2)
            overlay = np.clip(0.55 * base + 0.45 * coloured, 0, 1)
            axes[row, column].imshow(overlay)
            if column == 0:
                axes[row, column].set_ylabel(name, fontsize=9)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Ảnh đầu vào\n(stretch 224×224)", fontsize=9)

    FINAL_XAI.mkdir(parents=True, exist_ok=True)
    artifact_figure = FINAL_XAI / "fig_gradcam_final.pdf"
    figure.savefig(artifact_figure)
    shutil.copyfile(artifact_figure, OUT / "fig_gradcam_final.pdf")
    plt.close(figure)

    pd.DataFrame(cases).to_csv(FINAL_XAI / "gradcam_case_manifest.csv", index=False)
    pd.DataFrame(resnet_records + densenet_records).to_csv(
        FINAL_XAI / "gradcam_member_metrics.csv", index=False
    )

    consensus_rows = []
    for architecture, maps in (
        ("resnet18", resnet_maps),
        ("densenet121", densenet_maps),
    ):
        median = np.median(maps, axis=0)
        dispersion = np.percentile(maps, 75, axis=0) - np.percentile(maps, 25, axis=0)
        for case, cam, spread in zip(cases, median, dispersion):
            consensus_rows.append(
                {
                    "architecture": architecture,
                    "category": case["category"],
                    "group_id": case["group_id"],
                    "filename": case["filename"],
                    "consensus_border_mass_15pct": _border_mass(cam),
                    "mean_pixelwise_iqr": float(spread.mean()),
                }
            )
    pd.DataFrame(consensus_rows).to_csv(
        FINAL_XAI / "gradcam_consensus_metrics.csv", index=False
    )
    print(
        "Grad-CAM đồng thuận: "
        f"{OUT / 'fig_gradcam_final.pdf'}; manifest/metrics → {FINAL_XAI}/"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    oof, bench = load()
    manifest = pd.read_csv(
        V5 / "manifest_fold0.csv",
        usecols=["filename", "split_original", "class_name", "class_id"],
    )
    features = pd.read_csv(
        V4 / "nuisance_feature_manifest.csv",
        usecols=["filename", "file_size_per_pixel", "noise_estimate"],
    )

    figure_acquisition(manifest, features)
    figure_ladder(oof, bench)
    figure_saturation(oof, bench)
    figure_transitions(oof, bench)
    figure_hsas()
    figure_gradcam(oof, bench)

    produced = sorted(OUT.glob("fig_*.pdf"))
    print(f"{len(produced)} hình → {OUT}/")
    for path in produced:
        print(f"  {path.name}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
