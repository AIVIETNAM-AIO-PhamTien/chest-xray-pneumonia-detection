from pathlib import Path

import torch
from PIL import Image

from src.data.build_loaders import DataLoaderConfig, build_dataloaders
from src.data.imbalance import FocalLoss
from src.data.prepare import build_manifest, infer_patient_id
from src.data.split import create_splits


def _make_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (48, 64), color=value)
    image.save(path)


def _make_fake_dataset(root: Path) -> None:
    counts = {
        "train": {"NORMAL": 10, "PNEUMONIA": 16},
        "val": {"NORMAL": 4, "PNEUMONIA": 4},
        "test": {"NORMAL": 6, "PNEUMONIA": 8},
    }
    counter = 0
    for split_name, class_counts in counts.items():
        for class_name, count in class_counts.items():
            for index in range(count):
                counter += 1
                if class_name == "PNEUMONIA":
                    filename = f"person{counter}_bacteria_{index}.jpeg"
                else:
                    filename = f"IM-{counter:04d}-0001.jpeg"
                _make_image(
                    root / split_name / class_name / filename,
                    value=(counter * 7) % 255,
                )


def test_end_to_end_data_pipeline(tmp_path: Path) -> None:
    dataset_root = tmp_path / "chest_xray"
    _make_fake_dataset(dataset_root)

    manifest_path = tmp_path / "manifest.csv"
    manifest = build_manifest(
        input_root=dataset_root,
        output_csv=manifest_path,
        verify_images=True,
        compute_hash=True,
    )
    assert set(manifest["source_split"].unique()) == {"train", "val", "test"}

    split_df = create_splits(
        manifest=manifest,
        strategy="official",
        val_size=0.20,
        seed=7,
        group_col="patient_id",
    )
    split_path = tmp_path / "splits.csv"
    split_df.to_csv(split_path, index=False)

    bundle = build_dataloaders(
        split_csv=split_path,
        root_dir=dataset_root,
        config=DataLoaderConfig(
            batch_size=4,
            img_size=32,
            channels=3,
            augmentation_policy="none",
            balance_strategy="weighted_ce",
            num_workers=0,
            seed=7,
            verify_paths=True,
        ),
    )

    images, labels = next(iter(bundle.train))
    assert images.ndim == 4
    assert images.shape[1:] == (3, 32, 32)
    assert labels.dtype == torch.long
    assert bundle.val is not None
    assert bundle.test is not None
    assert bundle.class_counts.numel() == 2


def test_focal_loss_is_finite() -> None:
    criterion = FocalLoss(alpha=[1.2, 0.8], gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_pneumonia_patient_id_keeps_subtype() -> None:
    """Bacteria/virus counters overlap and must not be merged."""
    bacteria = infer_patient_id("person1_bacteria_1.jpeg")
    virus = infer_patient_id("person1_virus_6.jpeg")

    assert bacteria == "person1_bacteria"
    assert virus == "person1_virus"
    assert bacteria != virus
