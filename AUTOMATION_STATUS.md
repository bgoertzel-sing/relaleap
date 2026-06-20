# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, local paired ordinary-vs-pinned HEP comparison, support-instability diagnostics that expose whether ordinary repicked settling changes column support and whether pinned-vs-repicked logits diverge, an opt-in HEP support-stress config that intentionally produces nonzero support repicking without changing the checked Phase 0 baseline, a paired pinned-support support-stress comparison path with run-level support diagnostics in comparison artifacts, successful real-Chrome Colab evidence for the paired support-stress artifact run, a command-driven local pinned-support decision report that keeps pinned support opt-in under the current evidence, an opt-in clipped HEP settling mechanism probe for support-stress diagnostics, successful real-Chrome Colab evidence for the clipped support-stress HEP comparison, and a command-driven clipped-HEP decision report that keeps clipping opt-in under the current evidence.

## Latest Run

- Status: pass, 2026-06-19 bounded local clipped-HEP decision-report run completed.
- Changed: extended the command-driven decision report harness with `--report clipped-hep`, added clipped-HEP decision policy/tests/docs, and wrote the local ignored evidence report under `results/reports/clipped_hep_decision`.
- Verified: `./.venv-conda/bin/python -m relaleap.experiments.decision_report --report clipped-hep --comparison-dir results/comparisons/colab_support_stress_clipped_hep --artifact-check results/comparisons/colab_support_stress_clipped_hep/artifact_check_local.json --out results/reports/clipped_hep_decision` passed and returned `keep_opt_in`; `./.venv-conda/bin/python -m unittest tests.test_decision_report tests.test_check_artifacts` passed 18 tests; `./.venv-conda/bin/python -m unittest discover tests` passed 46 tests.
- Evidence: the clipped decision report reused passing Colab artifact evidence, observed `support_change_fraction` `0.7890625`, measured alpha `1.0` pinned-vs-repicked logit delta reduction from `3.3259902000427246` to `0.0023527145385742188`, and kept clipped HEP opt-in because every clipped nonzero alpha had `loss_improvement_from_alpha0` `0.0`.
- Blockers: none.
- Next step: design the next opt-in HEP mechanism probe that can improve loss under support stress while preserving the clipped settling stability budget.
