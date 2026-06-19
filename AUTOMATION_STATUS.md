# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, a local/Colab baseline gate, post-run artifact inspection, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run at 2026-06-19T11:09:34Z
- Changed: added `python -m relaleap.experiments.check_artifacts`, a command-driven post-run checker for completed comparison artifact trees. It verifies required `summary.json`, `metrics.csv`, and `notes.md` files for the comparison and child runs, summarizes comparison/verdict status, Phase 0 invariants, accepted HEP alpha, and optional baseline comparison status, and can fail if `baseline_comparison.json` is missing or failed. Updated `README.md` and `docs/colab_bridge.md` with the exact post-Colab inspection command.
- Verified: `git diff --check`; `python -m compileall relaleap tests tools`; `python -m unittest tests.test_smoke tests.test_compare tests.test_check_artifacts` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare tests.test_check_artifacts` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-artifact-check-comparison --baseline-reference baselines/phase0_char_smoke_comparison.json` passed; `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir /tmp/relaleap-artifact-check-comparison --require-baseline-comparison --out /tmp/relaleap-artifact-check-comparison/artifact_check.json` passed with checker `status: pass`, comparison `status: ok`, verdict `status: pass`, baseline comparison `status: pass`, 12 Phase 0 invariants checked/passed, accepted HEP alpha `0.25`, and no artifact failures.
- Blockers: live Colab/GPU execution was not completed. The local headless bridge attempt failed before opening Colab because Chromium was blocked by macOS sandbox state (`TargetClosedError` after `MachPortRendezvousServer ... Permission denied`). Existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution still succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: manually open the GitHub-backed Colab notebook, choose a GPU runtime, run all cells, then run `python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --require-baseline-comparison`.
