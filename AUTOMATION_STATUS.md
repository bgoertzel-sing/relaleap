# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, a local/Colab baseline gate, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run at 2026-06-19T10:11:20Z
- Changed: added `--baseline-reference` to the command-driven comparison harness; it writes `baseline_comparison.json` and exits nonzero if the stable Phase 0 fields or accepted HEP alpha diverge from `baselines/phase0_char_smoke_comparison.json`. Updated the Colab notebook to run the three-way comparison plus baseline gate, refreshed `docs/colab_bridge.md`/`README.md`, added tests for passing and drifting baseline comparisons, and made the Playwright Colab bridge fail with a concise blocker message instead of a traceback.
- Verified: `python -m compileall relaleap tests tools`; `python -m unittest tests.test_compare`; `git diff --check`; `python -m unittest tests.test_smoke tests.test_compare` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-colab-baseline-gate-final --baseline-reference baselines/phase0_char_smoke_comparison.json` passed with comparison `status: ok`, verdict `status: pass`, baseline comparison `status: pass`, 12 Phase 0 invariants checked/passed, accepted HEP alpha `0.25`, accepted delta `0.05185753`, and no baseline mismatches.
- Blockers: live Colab/GPU execution was not completed. The local headless bridge attempt failed before opening Colab because Chromium was blocked by macOS sandbox state (`TargetClosedError` after `MachPortRendezvousServer ... Permission denied`). Existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: manually open the GitHub-backed Colab notebook, choose a GPU runtime, run all cells, and inspect `results/comparisons/colab_phase0/baseline_comparison.json` for `status: pass`.
