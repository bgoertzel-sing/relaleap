# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, local paired ordinary-vs-pinned HEP comparison, support-instability diagnostics that expose whether ordinary repicked settling changes column support and whether pinned-vs-repicked logits diverge, and an opt-in HEP support-stress config that intentionally produces nonzero support repicking without changing the checked Phase 0 baseline.

## Latest Run

- Status: ok, 2026-06-19 bounded support-stress config increment completed
- Changed: added `configs/char_smoke_hep_support_stress.yaml`, a `model.columns.support_stress` opt-in in `relaleap/smoke.py`, a `support_stress` summary/notes field, focused unit coverage for nonzero config-level support repicking, and README documentation for the diagnostic-only stress run.
- Verified: `./.venv-conda/bin/python -m unittest tests.test_smoke.Phase0SmokeTest.test_support_stress_config_records_nonzero_repicking` passed; `./.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke_hep_support_stress.yaml --out results/runs/char_smoke_hep_support_stress` completed with `status: ok`, support-change fraction `0.7890625`, and pinned-vs-repicked logit delta up to `3.3259902000427246`; `./.venv-conda/bin/python -m unittest discover -s tests` passed 38 tests; `./.venv-conda/bin/python -m relaleap.experiments.compare --out results/comparisons/phase0_baseline_gate_after_support_stress --baseline-reference baselines/phase0_char_smoke_comparison.json` passed the checked schema v3 baseline with accepted HEP alpha `0.25`; `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/phase0_baseline_gate_after_support_stress --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison` returned `status: pass`.
- Blockers: none. Generated `results/` artifacts remain ignored by git, so the tracked record is this status file plus the local artifact tree.
- Next step: add a paired pinned-support variant of the support-stress config and compare ordinary-stress versus pinned-stress artifacts before changing the checked Phase 0 baseline.
