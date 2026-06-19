# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, real smoke metrics, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: replaced synthetic `smoke_loss` rows with actual Phase 0 base, zero-init, pre-step residual, and post-step residual losses; `summary.json`, `metrics.csv`, and `notes.md` now report the same real smoke loss stream; tests pin the new metric schema.
- Verified: `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke` passed with 2 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke` passed; `.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke.yaml --out /tmp/relaleap-phase0-real-metrics-final` passed with all Phase 0 invariants true and required artifacts present.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add a tiny configurable residual-training loop on the char smoke batch so `max_steps` controls real residual updates and `metrics.csv` records one row per update.
