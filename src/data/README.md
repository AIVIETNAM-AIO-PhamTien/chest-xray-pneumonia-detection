# CXR data pipeline

This package covers local dataset indexing, image validation, split creation,
augmentation, class-imbalance handling, and PyTorch DataLoaders.

It does **not** silently download the dataset. Download the Kaggle dataset first:

```bash
python -m pip install kaggle
./scripts/download_data.sh data/raw
```

The Kaggle CLI needs valid Kaggle credentials. Do not commit `kaggle.json`.

## Files

```text
src/data/
  cxr_dataset.py    Robust local image reader
  transforms.py     none / paper / advanced augmentation policies
  imbalance.py      sampler, class weights, effective weights, focal loss
  prepare.py        Folder scanner and manifest creator
  split.py          official, paper-style, or all-data split strategies
  build_loaders.py  End-to-end Dataset + DataLoader + criterion builder
```

## Recommended workflow

### 1. Build a manifest

```bash
python -m src.data.prepare \
  --input-root data/raw \
  --output data/processed/cxr_manifest.csv \
  --compute-hash
```

This scans the downloaded folders, validates images, stores relative paths,
extracts a best-effort patient ID, and optionally hashes files to detect exact
duplicates.

### 2. Create splits

Recommended trustworthy setup: keep the supplied test folder untouched and
create a new validation set from the original train+val pool.

```bash
python -m src.data.split \
  --input data/processed/cxr_manifest.csv \
  --output data/processed/cxr_splits.csv \
  --strategy official \
  --val-size 0.10 \
  --seed 42
```

Paper-style 80/20 image-level reproduction using original train+val only:

```bash
python -m src.data.split \
  --input data/processed/cxr_manifest.csv \
  --output data/processed/cxr_paper_splits.csv \
  --strategy paper \
  --test-size 0.20 \
  --val-size 0 \
  --group-col "" \
  --seed 42
```

### 3. Smoke-test DataLoaders

```bash
python -m src.data.build_loaders \
  --csv data/processed/cxr_splits.csv \
  --root-dir data/raw/chest_xray \
  --augmentation advanced \
  --balance weighted_ce \
  --channels 3 \
  --batch-size 32 \
  --num-workers 4 \
  --verify-paths
```

If the archive contains another nested `chest_xray/` folder, point `--root-dir`
to the directory that directly contains `train/`, `val/`, and `test/`.

## Training integration

```python
from pathlib import Path

from src.data.build_loaders import DataLoaderConfig, build_dataloaders

bundle = build_dataloaders(
    split_csv=Path("data/processed/cxr_splits.csv"),
    root_dir=Path("data/raw/chest_xray"),
    config=DataLoaderConfig(
        batch_size=32,
        img_size=128,
        channels=3,
        augmentation_policy="advanced",
        balance_strategy="weighted_ce",
        num_workers=4,
        pretrained=False,
    ),
)

train_loader = bundle.train
val_loader = bundle.val
test_loader = bundle.test
criterion = bundle.criterion.to(device)
```

## Choosing an imbalance strategy

Use one primary correction at a time:

- `weighted_ce`: strong default for this binary dataset.
- `sampler`: balances sampled training examples; loss stays unweighted.
- `focal`: emphasizes hard examples and includes inverse-frequency alpha.
- `effective_ce` / `effective_focal`: softer class weighting for larger skew.
- `none`: useful for an ablation baseline.

Do not apply SMOTE directly to flattened X-ray pixels in this pipeline. Pixel
interpolation can create unrealistic anatomy, consumes substantial memory, and
can leak information if performed before splitting. If SMOTE is required for an
experiment, apply it only to training-set embeddings in a separate feature-based
classifier and report that as a distinct experiment.

## Augmentation policies

- `paper`: 128x128 resize, horizontal flip, zoom 0.9-1.1, brightness +/-20%.
- `advanced`: adds conservative rotation, translation, contrast, blur, and mild
  Gaussian/speckle noise while avoiding vertical flips and aggressive crops.
- `none`: deterministic resize and normalization.

Validation and test transforms are always deterministic.

## Tests

```bash
pytest -q
```
