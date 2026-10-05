#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT_PATH OUTPUT_JSON" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" -m cmfo evaluate \
  --config configs/train/e1_cmfo.yaml \
  --checkpoint "$1" \
  --output "$2" \
  --device auto

