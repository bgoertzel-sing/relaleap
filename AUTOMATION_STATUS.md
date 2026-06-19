# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run
- Changed: added a compact comparison `verdict` to aggregate `summary.json` and `notes.md`, including comparison pass/fail status, Phase 0 invariant counts/failures, and best HEP alpha by loss; documented the verdict in `README.md`; added focused comparison tests for passing and failing verdicts.
- Verified: `git diff --check`; `python -m compileall relaleap tests`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-comparison-verdict-1781852877` passed with comparison `status: ok`, verdict `status: pass`, 12 Phase 0 invariants checked/passed, and best HEP alpha `1.0` by loss `3.52012658`.
- Blockers: existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: add an explicit HEP alpha acceptance policy to the comparison verdict so loss improvement is weighed against ordinary-logit delta before GPU runs.
