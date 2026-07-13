#!/usr/bin/env bash
# Push a Kaggle kernel for one experiment.
#
# Convention: 1 experiment/config = 1 kernel slug, so two people never push
# over the same kernel. Requires ~/.kaggle/kaggle.json (never commit it).
#
# Usage:
#   kaggle/push_kernel.sh <kernel-dir>
#
# <kernel-dir> must contain a kernel-metadata.json (see Kaggle API docs:
# https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kernel-metadata).
# Naming convention for kernel slugs: pneumo-<block>-<expname>-v<N>

set -euo pipefail

KERNEL_DIR="${1:?Usage: kaggle/push_kernel.sh <kernel-dir>}"

uv run kaggle kernels push -p "$KERNEL_DIR"
