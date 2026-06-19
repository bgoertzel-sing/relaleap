# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: added `configs/char_smoke_hep.yaml` to the default comparison preset, so `python -m relaleap.experiments.compare` now produces supervised, PC, and HEP run artifacts in one local command; propagated HEP alpha sweep fields into aggregate `summary.json`, `metrics.csv`, and `notes.md`; updated comparison tests and README wording.
- Verified: `git diff --check`; `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-comparison-three-way-1781849222` passed with all three runs `ok`, all Phase 0 invariants true, and HEP CE losses `3.56317067`, `3.55195642`, `3.54104066`, `3.52012658` for alpha `0.0`, `0.25`, `0.5`, `1.0`.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add a compact comparison verdict field that highlights invariant pass/fail plus best HEP alpha by loss.
