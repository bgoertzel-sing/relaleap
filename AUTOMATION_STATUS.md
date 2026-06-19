# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run at 2026-06-19T09:10:12Z
- Changed: added `--baseline-out` to the command-driven comparison harness; generated `baselines/phase0_char_smoke_comparison.json` as a compact stable Phase 0 baseline; documented the refresh command in `README.md`; added tests for the baseline writer and checked-in baseline contract.
- Verified: `python -m compileall relaleap tests`; `python -m unittest tests.test_compare`; `git diff --check`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-phase0-baseline-generate --baseline-out baselines/phase0_char_smoke_comparison.json` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-baseline-cli-verify --baseline-out /tmp/relaleap-baseline-cli-verify/phase0_baseline.json` passed with comparison `status: ok`, verdict `status: pass`, 12 Phase 0 invariants checked/passed, best HEP alpha by loss `1.0`, accepted HEP alpha `0.25`, accepted delta `0.05185753`, and 2 rejected nonzero alpha candidates.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: run the same three-way comparison command in Colab/GPU and compare the accepted HEP alpha against the checked-in local baseline.
