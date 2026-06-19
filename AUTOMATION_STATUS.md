# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 wrap-up and handoff: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, and a concise Phase 0 report.

## Latest Run

- Status: ok, 2026-06-19 bounded documentation/reporting run completed
- Changed: added `docs/phase0_report.md`, summarizing the checked schema v3 local baseline and checked Colab artifact tree, including Phase 0 invariant counts, artifact-contract evidence, loss trajectories, HEP alpha acceptance, and the narrow handoff recommendation.
- Verified: `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out results/comparisons/colab_phase0/artifact_check.json`; `./.venv-conda/bin/python -m unittest discover -s tests`. The checker reported status `pass`, 12/12 Phase 0 invariants passing, 9/9 artifact invariants passing, baseline comparison `pass`, baseline reference comparison `pass`, and accepted HEP alpha `0.25`; unittest ran 34 tests passing. `pytest` was not installed in `.venv-conda`, so the stdlib runner was used.
- Blockers: none. Colab artifacts and bridge evidence remain present locally under ignored `results/` paths and are not tracked by git.
- Next step: add pinned support to the residual-column experiment while preserving the Phase 0 comparison and artifact contracts as regression gates.
