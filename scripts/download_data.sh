#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-data/raw}"
DATASET="paultimothymooney/chest-xray-pneumonia"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "The Kaggle CLI is not installed. Run: python -m pip install kaggle" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
kaggle datasets download -d "$DATASET" -p "$OUTPUT_DIR" --unzip

echo "Dataset extracted below: $OUTPUT_DIR"
echo "Keep Kaggle credentials outside the repository."
