# CXR data pipeline (src/train.py's single-holdout CLI baseline)

This package is the DataLoader/imbalance-handling half of `src/train.py`'s
historical single-holdout CLI baseline. Per CLAUDE.md, `src.train` is not used
to produce the final 5-fold report numbers -- that pipeline lives in
`src/dataset.py` + `src/transforms.py` + `src/splits.py` instead (used by the
canonical `notebooks/chest_xray_research_complete.ipynb` and the report/audit
scripts). See the module docstring in `src/data/__init__.py`.

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
  build_loaders.py  End-to-end Dataset + DataLoader + criterion builder
```

Manifest building and train/val/test split assignment for this baseline are
handled by `src/splits.py` (`build_manifest`, `make_splits`), the same module
the canonical notebook uses -- `src/train.py` calls it directly and writes the
resulting split CSV before handing it to `build_loaders.build_dataloaders`
below. There is no standalone CLI step for this anymore; run
`python -m src.train --config configs/protocol_b.yaml` to exercise the whole
chain end to end.

## Recommended workflow

### 1. Build a manifest + split CSV with src/splits.py

`src/splits.py` has no CLI of its own; call it directly (this is exactly what
`src/train.py` does internally):

```python
from src.dataset import find_data_root
from src.splits import build_manifest, make_splits, save_manifest

root = find_data_root("data/raw")
manifest = make_splits(build_manifest(root), protocol="b_patient_grouped", seed=42)
save_manifest(
    manifest[["path", "class_id", "split"]].rename(columns={"class_id": "label"}),
    "data/processed/cxr_splits.csv",
)
```

### 2. Smoke-test DataLoaders

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
