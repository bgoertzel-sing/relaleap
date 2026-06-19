# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, and a local paired ordinary-vs-pinned HEP comparison showing pinned support does not yet need a separate checked baseline.

## Latest Run

- Status: ok, 2026-06-19 bounded paired local HEP comparison completed
- Changed: ran the existing command-driven comparison harness for ordinary HEP versus pinned-support HEP with `./.venv-conda/bin/python -m relaleap.experiments.compare --config configs/char_smoke_hep.yaml --config configs/char_smoke_pinned_hep.yaml --out results/comparisons/pinned_vs_ordinary_hep`; documented the paired comparison command and decision in `README.md`.
- Verified: `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/pinned_vs_ordinary_hep` returned `status: pass`, comparison verdict `pass`, 8/8 Phase 0 invariants passing, 6/6 artifact invariants passing, and accepted HEP alpha `0.25`; the ordinary and pinned HEP alpha sweeps were identical for alpha `0.0`, `0.25`, `0.5`, and `1.0`, so pinned support remains an opt-in artifact-only smoke path rather than a separate checked baseline.
- Blockers: none. Generated `results/` artifacts remain ignored by git, so the tracked record is this status file plus the local artifact tree.
- Next step: start the next methodology increment by adding a support-instability diagnostic that can distinguish ordinary repicked settling from pinned-support settling when column support changes across HEP steps.
