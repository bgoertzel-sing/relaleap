# RelaLeap

RelaLeap is an experimental codebase for residual layer learning prototypes, beginning with columnar PC/CC residual layers on frozen backpropagation-trained transformer bases.

The first goal is not frontier-scale performance. The first goal is a reproducible small-scale methodology for testing:

- frozen-base residual adaptation;
- sparse columnar rank-one residual atoms;
- predictive-coding residual training;
- highway error propagation;
- windowed settling and pinned support;
- later causal-coding audits and simple symbolic heads.

## Initial Mission

Validate a minimal experimental harness on char-level Tiny Shakespeare before moving to larger GPU experiments.

## First Campaign

Run only infrastructure and early residual/HEP experiments first:

1. Phase 0 invariants.
2. Tiny BP base: 2 layers, hidden dim 32, char vocabulary, sequence length 32.
3. Freeze base.
4. One-site columns: 8 columns, 4 atoms per column, top-k 1.
5. PC residual training.
6. HEP alpha sweep at small inference-step counts.
7. Report before adding pinned support, causal-coding mechanisms, or symbolic heads.

## Early Guardrails

- Zero-initialized columns must preserve base logits within tolerance.
- Frozen base parameters must not change during residual-layer training.
- HEP with alpha 0 must match ordinary inference.
- Every run must produce `summary.json`, `metrics.csv`, and `notes.md`.
- Stop after three infrastructure failures or any invariant failure.

## Execution Model

Every experiment should be runnable as a plain Python command so it can run locally, in Colab, or on another GPU backend:

```bash
python -m relaleap.experiments.run --config configs/char_smoke.yaml
```

The current supervised/PC/HEP char smoke comparison is also command-driven:

```bash
python -m relaleap.experiments.compare
```

The comparison `summary.json` includes a compact verdict with aggregate
Phase 0 invariant pass/fail status, required artifact pass/fail status, the
best HEP alpha by loss, and an HEP alpha acceptance policy. By default,
accepted nonzero HEP alphas must improve loss over alpha 0 while keeping
ordinary-logit delta at or below `0.1`; tune that gate with
`--hep-max-logit-delta` and `--hep-min-loss-improvement`.
To refresh the checked-in compact Phase 0 baseline after an intentional
methodology change, run:

```bash
python -m relaleap.experiments.compare --baseline-out baselines/phase0_char_smoke_comparison.json
```

The current baseline records all 12 Phase 0 model invariants and all 9 child-run
artifact invariants passing, with an accepted HEP alpha of `0.25` under the
default logit-delta policy.

To compare a fresh local or Colab/GPU run against that baseline, run:

```bash
python -m relaleap.experiments.compare --out results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

This writes `baseline_comparison.json` under the comparison output directory
and exits nonzero if the accepted HEP alpha, Phase 0 invariant result, artifact
contract, or config set diverges from the checked-in baseline.

To inspect a completed local or Colab comparison artifact tree without rerunning
experiments, run:

```bash
python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

This checks the required `summary.json`, `metrics.csv`, and `notes.md` artifacts
for the comparison and child runs, then reports the verdict, Phase 0 invariant
status, artifact invariant status, accepted HEP alpha, and whether the completed
summary still matches the checked-in local baseline. Add
`--require-baseline-comparison` when the run was expected to write
`baseline_comparison.json` during execution.

The current tiny HEP alpha sweep is also command-driven:

```bash
python -m relaleap.experiments.run --config configs/char_smoke_hep.yaml --out results/runs/char_smoke_hep
```

Colab should be treated as a temporary GPU runner, not the source of truth. The GitHub repo, config files, and run artifacts are the source of truth.

## Temporary Colab Bridge

The first GitHub-backed Colab notebook is:

```text
https://colab.research.google.com/github/bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb
```

For a brittle command-line/browser automation bridge, see:

```text
docs/colab_bridge.md
tools/colab_playwright_runner.py
```

Use a modern Python environment for that bridge. The old conda `base` Python
3.7 environment cannot install current Playwright wheels.

One-command setup:

```bash
bash tools/setup_colab_bridge.sh
```
