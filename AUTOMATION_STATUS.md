# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective comparison reports, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: added `python -m relaleap.experiments.compare`, which runs the supervised and PC char-smoke configs into sibling directories and writes top-level `summary.json`, `metrics.csv`, and `notes.md` comparison artifacts with initial/final loss, delta, and ratio per objective; documented the command in `README.md`.
- Verified: `git diff --check`; `python -m compileall relaleap tests`; `python -m unittest tests.test_compare tests.test_smoke` passed with 3 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_compare tests.test_smoke` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-phase0-objective-comparison` passed with both objectives `ok`, all Phase 0 invariants true, supervised CE loss `3.61089253 -> 3.56317067`, and PC logit MSE loss `0.02878798 -> 0.02869168`.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add a tiny HEP alpha sweep config and implementation path for `hep_alpha > 0`, preserving the current alpha-0 equivalence invariant as the baseline.
