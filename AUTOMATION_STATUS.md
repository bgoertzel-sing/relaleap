# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, local paired ordinary-vs-pinned HEP comparison, and support-instability diagnostics that expose whether ordinary repicked settling changes column support and whether pinned-vs-repicked logits diverge.

## Latest Run

- Status: ok, 2026-06-19 bounded support-instability diagnostic increment completed
- Changed: added a command-driven HEP support-instability diagnostic to `relaleap/smoke.py`, propagated `hep_support_change_fraction` and `hep_pinned_vs_repicked_logit_delta` through run/comparison CSVs and notes, added focused unit coverage for a controlled repick-vs-pinned case, and documented the current paired HEP interpretation in `README.md`.
- Verified: `./.venv-conda/bin/python -m unittest discover -s tests` passed 37 tests; `./.venv-conda/bin/python -m relaleap.experiments.compare --config configs/char_smoke_hep.yaml --config configs/char_smoke_pinned_hep.yaml --out results/comparisons/pinned_vs_ordinary_hep` completed with verdict `pass`; `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/pinned_vs_ordinary_hep` returned `status: pass`; `./.venv-conda/bin/python -m relaleap.experiments.compare --out results/comparisons/phase0_baseline_gate_after_support_diag --baseline-reference baselines/phase0_char_smoke_comparison.json` passed the checked schema v3 baseline; `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/phase0_baseline_gate_after_support_diag --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison` returned `status: pass` with baseline comparison `pass`. The current HEP smoke still reports support-change fraction `0.0` and pinned-vs-repicked logit delta `0.0` for every swept alpha.
- Blockers: none. Generated `results/` artifacts remain ignored by git, so the tracked record is this status file plus the local artifact tree.
- Next step: add a small opt-in HEP stress config that intentionally produces nonzero support repicking so the new diagnostic can guide pinned-support methodology before changing the checked Phase 0 baseline.
