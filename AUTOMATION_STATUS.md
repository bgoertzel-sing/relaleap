# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, local paired ordinary-vs-pinned HEP comparison, support-instability diagnostics that expose whether ordinary repicked settling changes column support and whether pinned-vs-repicked logits diverge, an opt-in HEP support-stress config that intentionally produces nonzero support repicking without changing the checked Phase 0 baseline, a paired pinned-support support-stress comparison path with run-level support diagnostics in comparison artifacts, successful real-Chrome Colab evidence for the paired support-stress artifact run, and a command-driven local pinned-support decision report that keeps pinned support opt-in under the current evidence.

## Latest Run

- Status: pass, 2026-06-19 bounded local decision-report run completed from extracted Colab support-stress artifacts.
- Changed: added `relaleap.experiments.decision_report`, focused unit coverage, README usage notes, and generated ignored local evidence under `results/reports/pinned_support_decision/`.
- Verified: `./.venv-conda/bin/python -m relaleap.experiments.decision_report --comparison-dir results/comparisons/colab_support_stress_pinned_vs_repicked --artifact-check results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check_local.json --out results/reports/pinned_support_decision` passed with decision `keep_opt_in`; `./.venv-conda/bin/python -m unittest tests.test_decision_report tests.test_check_artifacts tests.test_compare` passed 29 tests; `./.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare tests.test_check_artifacts tests.test_colab_playwright_runner tests.test_decision_report` passed 41 tests.
- Evidence: the report consumed the passing support-stress artifact check and comparison summary, recorded `support_change_fraction` `0.7890625` and `pinned_vs_repicked_logit_delta` `3.3259902000427246`, and refused promotion because pinned nonzero HEP alphas had negative loss improvement from alpha 0 and exceeded the default `0.1` ordinary-logit delta budget.
- Blockers: none.
- Next step: start the next HEP mechanism experiment while keeping pinned support as an opt-in diagnostic.
