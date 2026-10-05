#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${CMFO_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
CHECKPOINT_ROOT="${CMFO_CHECKPOINTS_ROOT:-/root/autodl-tmp/CMFO/checkpoints}"
RUN_ID="${RUN_ID:-cmfo_e1_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "${REPO_ROOT}"
nvidia-smi
PYTHONPATH=src "${PYTHON_BIN}" -m cmfo train \
  --config configs/train/e1_cmfo.yaml \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --run-id "${RUN_ID}" \
  --device auto

