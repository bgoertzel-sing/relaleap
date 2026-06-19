# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 real-Chrome Colab Phase 0 bridge completed with hardened rendered-output evidence
- Changed: no code changes in this bounded run. Refreshed the ignored local evidence file `results/colab_bridge_evidence/latest_colab_output.txt` and debug snapshot `results/colab_bridge_debug/20260619T192356Z_after_run_all.{html,png}` via the real-Chrome CDP bridge.
- Verified: `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot` passed. The rendered Colab evidence reports Tesla T4 GPU, `torch: 2.11.0+cu128`, `cuda_available: True`, comparison verdict `"status": "pass"`, baseline comparison `"status": "pass"` with no mismatches, and accepted HEP alpha `0.25` with loss improvement `0.011214256286621094`.
- Blockers: no blocker for this bounded Colab evidence step. Colab artifacts remain in the notebook runtime unless copied back or mounted locally; the preserved local evidence is the rendered-output capture and debug snapshot.
- Next step: copy or mount `results/comparisons/colab_phase0` from the Colab runtime locally, then run `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out results/comparisons/colab_phase0/artifact_check.json`.
