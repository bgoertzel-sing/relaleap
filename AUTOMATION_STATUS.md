# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: implemented a bounded HEP alpha > 0 smoke path in `forward_with_hep_alpha`, preserving alpha-0 ordinary-inference equivalence after residual training; added `configs/char_smoke_hep.yaml`; wrote HEP sweep metrics into `summary.json`, `metrics.csv`, and `notes.md`; documented the command in `README.md`.
- Verified: `git diff --check`; `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke_hep.yaml --out /tmp/relaleap-phase0-hep-sweep` passed with all Phase 0 invariants true, alpha 0 max ordinary-logit delta `0.0`, and HEP CE losses `3.56317067`, `3.55195642`, `3.54104066`, `3.52012658` for alpha `0.0`, `0.25`, `0.5`, `1.0`.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add the HEP sweep config to the comparison runner so supervised, PC, and HEP artifacts can be produced by one local command.
