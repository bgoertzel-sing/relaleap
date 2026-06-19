# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run at 2026-06-19T08:09:53Z
- Changed: added an explicit HEP alpha acceptance policy to the aggregate comparison verdict and notes; the policy requires nonzero alpha, loss improvement over alpha 0, and ordinary-logit delta at or below the default `0.1` budget, with CLI overrides `--hep-max-logit-delta` and `--hep-min-loss-improvement`; documented the policy in `README.md`; added tests proving the accepted alpha can differ from the raw best-by-loss alpha.
- Verified: `python -m compileall relaleap tests`; `python -m unittest tests.test_compare`; `git diff --check`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-comparison-hep-policy-clean` passed with comparison `status: ok`, verdict `status: pass`, 12 Phase 0 invariants checked/passed, best HEP alpha by loss `1.0`, accepted HEP alpha `0.25`, accepted delta `0.05185753`, and 2 rejected nonzero alpha candidates.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: save a checked-in Phase 0 comparison baseline artifact or fixture that records the current accepted HEP alpha before moving the same command to Colab/GPU.
