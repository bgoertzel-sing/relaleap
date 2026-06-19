# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Pinned-support handoff after Phase 0: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, and an opt-in pinned-support HEP smoke path.

## Latest Run

- Status: ok, 2026-06-19 bounded pinned-support implementation run completed
- Changed: added opt-in pinned support for HEP settling so the ordinary-pass top-k residual-column support can be reused across later settling updates; added `configs/char_smoke_pinned_hep.yaml`; surfaced `phase0.pinned_support` in run summaries/notes; documented the pinned-support command; added focused tests for support pinning and config reporting. The default Phase 0 comparison config set and checked baseline were intentionally left unchanged.
- Verified: `./.venv-conda/bin/python -m unittest discover -s tests` ran 36 tests passing; `./.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke_pinned_hep.yaml --out results/runs/char_smoke_pinned_hep` completed with status `ok`, `pinned_support: true`, 4/4 Phase 0 invariants passing, required artifacts passing, and accepted-equivalent alpha-0 HEP delta `0.0`; `./.venv-conda/bin/python -m relaleap.experiments.compare --out results/comparisons/local_phase0_pinned_support_guard --baseline-reference baselines/phase0_char_smoke_comparison.json` completed with baseline comparison status `pass`, 12/12 default Phase 0 invariants passing, 9/9 artifact invariants passing, and accepted HEP alpha `0.25`.
- Blockers: none. New local evidence remains under ignored `results/` paths and is not tracked by git. No Colab/GPU run was attempted in this bounded local implementation pass.
- Next step: run `configs/char_smoke_pinned_hep.yaml` through the real-Chrome Colab bridge or an equivalent GPU command runner and preserve the resulting artifact evidence.
