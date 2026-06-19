# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: added `training.residual_objective` with `supervised_ce` as the invariant baseline and `pc_logit_mse` as the first PC-style residual objective; added `configs/char_smoke_pc.yaml`; `summary.json`, `metrics.csv`, and `notes.md` now record the residual objective.
- Verified: `git diff --check`; `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke` passed with 3 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke` passed; `.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke.yaml --out /tmp/relaleap-phase0-objective-toggle-smoke` passed with 10 supervised residual updates, all Phase 0 invariants true, and required artifacts present; `.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke_pc.yaml --out /tmp/relaleap-phase0-pc-objective-smoke` passed with 10 PC residual updates, all Phase 0 invariants true, and required artifacts present.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add a compact comparison report that runs both char smoke configs and summarizes supervised-vs-PC loss trajectories from their `metrics.csv` artifacts.
