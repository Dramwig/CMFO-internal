#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${CMFO_OUTPUT_ROOT:-${REPO_ROOT}/outputs/smoke}"
CHECKPOINT_ROOT="${CMFO_CHECKPOINTS_ROOT:-/root/autodl-tmp/CMFO/checkpoints}"
RUN_ID="${RUN_ID:-cmfo_e1_cpu_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "${REPO_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" -m cmfo smoke \
  --config configs/smoke/e1_cpu.yaml \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --run-id "${RUN_ID}" \
  --device cpu

