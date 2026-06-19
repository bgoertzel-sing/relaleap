# Phase 0 Report

Date: 2026-06-19

## Scope

Phase 0 validated the command-driven RelaLeap smoke harness on Tiny Shakespeare char-level runs before adding pinned support, causal-coding mechanisms, or symbolic heads.

The checked comparison covers:

- `configs/char_smoke.yaml`: supervised CE residual update.
- `configs/char_smoke_pc.yaml`: PC-style logit MSE residual update.
- `configs/char_smoke_hep.yaml`: supervised CE residual update with HEP alpha sweep `0.0,0.25,0.5,1.0`.

All runs use a two-layer hidden-dim-32 base, sequence length 32, one insertion site, 8 residual columns, 4 atoms per column, top-k 1, seed 1, and 10 residual training steps.

## Evidence

The source-of-truth local baseline is:

```text
baselines/phase0_char_smoke_comparison.json
```

The checked Colab/GPU artifact tree is local, ignored by git, and was extracted from the rendered notebook bundle:

```text
results/comparisons/colab_phase0
results/colab_bridge_evidence/latest_colab_output.txt
```

The artifact tree was inspected with:

```bash
./.venv-conda/bin/python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_phase0 \
  --baseline-reference baselines/phase0_char_smoke_comparison.json \
  --require-baseline-comparison \
  --out results/comparisons/colab_phase0/artifact_check.json
```

Checker result:

- Overall status: `pass`.
- Comparison verdict: `pass`.
- Phase 0 model invariants: 12/12 passing.
- Artifact invariants: 9/9 passing.
- Written `baseline_comparison.json`: present and passing.
- Baseline reference comparison: passing with zero mismatches.

## Invariants

Each child run reported these model invariants as passing:

- Zero-initialized residual columns preserve base logits.
- Frozen base parameters remain unchanged during residual training.
- HEP alpha `0.0` matches ordinary inference.
- Residual parameters update during residual training.

Across the three child runs, the comparison summary therefore reports 12/12 Phase 0 model invariants passing.

Each child run also reported the required artifact contract as passing:

- `summary.json`
- `metrics.csv`
- `notes.md`

Across the three child runs, the comparison summary reports 9/9 artifact invariants passing.

## Loss Summary

The loss scales differ by objective, so each trajectory should be read against its own initial loss.

| Experiment | Objective | Initial residual loss | Final residual loss | Delta |
| --- | --- | ---: | ---: | ---: |
| `char_smoke` | `supervised_ce` | 3.61089253 | 3.56317067 | -0.04772186 |
| `char_smoke_pc` | `pc_logit_mse` | 0.02878798 | 0.02869168 | -0.00009630 |
| `char_smoke_hep` | `supervised_ce` | 3.61089253 | 3.56317067 | -0.04772186 |

## HEP Alpha Result

The HEP sweep in `char_smoke_hep` produced:

| Alpha | Loss | Max logit delta from ordinary | Gate result |
| ---: | ---: | ---: | --- |
| 0.0 | 3.5631706715 | 0.0000000000 | baseline |
| 0.25 | 3.5519564152 | 0.0518576503 | accepted |
| 0.5 | 3.5410408974 | 0.1037149131 | rejected: over `0.1` logit-delta budget |
| 1.0 | 3.5201268196 | 0.2074296474 | rejected: over `0.1` logit-delta budget |

The best raw loss was alpha `1.0`, but the acceptance policy requires improvement over alpha `0.0` while keeping ordinary-logit delta at or below `0.1`. Under that policy, the accepted nonzero HEP alpha is `0.25`, with loss improvement `0.0112142563` from alpha `0.0`.

## Conclusion

Phase 0 infrastructure is complete enough to use as the guardrail for the next research increment. The command-driven harness, baseline gate, Colab artifact extraction, after-the-fact checker, per-run artifact identity checks, and HEP alpha acceptance policy all passed against the checked local baseline and the checked Colab artifact tree.

The next research step should stay narrow: add pinned support to the residual-column experiment while preserving the Phase 0 comparison and artifact contracts as regression gates.
