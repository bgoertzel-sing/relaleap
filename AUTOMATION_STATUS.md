# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable residual smoke training, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: `run.max_steps` now controls real residual optimizer updates in the Phase 0 char smoke path; `metrics.csv` records step 0 plus one `residual_update` row per update, while `summary.json` and `notes.md` report `training_steps` and final residual loss.
- Verified: `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke` passed with 2 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke` passed; `.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke.yaml --out /tmp/relaleap-phase0-training-loop-smoke` passed with 10 residual updates, all Phase 0 invariants true, and required artifacts present.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add the first PC-style residual objective toggle, preserving the current supervised residual smoke loop as the invariant baseline.
