# CMFO Internal

Research code and evidence workspace for **Continuous Multimodal Field Operator**.

The first milestone is deliberately narrow: implement a small text–image field operator and test whether changing hidden quadrature grids preserves predictions better than compute-matched discrete baselines.

The repository includes the executable E0/E1 research system. It is an implementation and
evaluation harness, not evidence that the paper hypotheses are already true.

## Layout

    configs/                Reproducible data/model/train/eval/ablation configs.
    docs/                   Claims, experiment plans, records, and environment facts.
    scripts/                Thin data, training, evaluation, smoke, and sync entrypoints.
    src/cmfo/               Reusable Python package.
    tests/                  CPU unit tests, smoke tests, and integration tests.
    artifacts/              Small reports, figures, and tables safe for version control.

Large assets live outside the repository on pro6000:

    /root/autodl-tmp/CMFO/datasets
    /root/autodl-tmp/CMFO/weights
    /root/autodl-tmp/CMFO/checkpoints

## Implemented system

- deterministic text–image controlled scenes with arbitrary image observation grids;
- full UTF-8 byte observations and native 1D/2D coordinates;
- padded variable-node batching with explicit quadrature weights;
- continuous byte/image lifting;
- pointwise, local integral, low-rank global, and bidirectional cross-modal operators;
- global relation classification and coordinate-query segmentation heads;
- B0 patch Transformer, B1 point Transformer, B2 Perceiver, B3 fixed-grid CMFO, and
  B4 augmentation-matched Transformer baselines;
- multi-task training, discretization consistency loss, checkpointing, calibration,
  quadrature sweeps, field consistency, and latency reporting.

## Current research milestone

1. Complete the novelty matrix.
2. Freeze the controlled E1 scene and grid protocol.
3. Implement coordinate observations, quadrature utilities, and one minimal operator block.
4. Implement matched B0–B4 baselines.
5. Pass the E0 numerical and structural tests before real-data training.

See [docs/roadmap.md](docs/roadmap.md) for gates and [docs/experiment-plan.md](docs/experiment-plan.md) for the evaluation contract.

## Development

The package targets Python 3.10 or newer. The full ML dependency set is intentionally not frozen until the minimal architecture and server environment are selected.

For the current development environment:

    python -m pip install -e ".[dev]"
    python -m pytest

Run the bounded CPU end-to-end smoke test:

    PYTHON_BIN=/root/autodl-tmp/conda/envs/pf-vlm/bin/python \
    bash scripts/smoke/run_cpu_smoke.sh

Formal GPU experiments run from:

    ssh pro6000
    cd /root/autodl-tmp/CMFO/CMFO-internal

Read [AGENTS.md](AGENTS.md) before changing code or launching runs.
