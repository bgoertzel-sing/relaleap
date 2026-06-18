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

Colab should be treated as a temporary GPU runner, not the source of truth. The GitHub repo, config files, and run artifacts are the source of truth.

