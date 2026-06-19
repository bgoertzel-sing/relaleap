# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 per-run artifact baseline schema v3
- Changed: promoted per-run child artifact invariant status into the stable comparison baseline and baseline-reference mismatch checks. Regenerated `baselines/phase0_char_smoke_comparison.json` as schema v3 with each child run pinned to passing `summary_json`, `metrics_csv`, and `notes_md` contracts, while preserving accepted HEP alpha `0.25`.
- Verified: `python -m unittest tests.test_compare tests.test_check_artifacts tests.test_smoke` passed with 4 torch skips; `.venv-conda/bin/python -m unittest tests.test_compare tests.test_check_artifacts tests.test_smoke` passed; `python -m compileall relaleap tests tools` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-run-artifact-baseline-v3 --baseline-reference baselines/phase0_char_smoke_comparison.json` passed with accepted HEP alpha `0.25`; `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir /tmp/relaleap-run-artifact-baseline-v3 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out /tmp/relaleap-run-artifact-baseline-v3/artifact_check_latest.json` passed with zero failures.
- Blockers: live Colab/GPU execution is blocked in this automation sandbox because process inspection and localhost CDP probing are denied (`ps` returned operation not permitted; `http://127.0.0.1:9222/json/version` returned `Operation not permitted`). Recovery action: run Chrome with `--remote-debugging-port=9222` in a context that permits localhost CDP access, logged into Google/Colab, then run the documented `tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot`.
- Next step: run the real-Chrome CDP Colab bridge from a localhost-CDP-enabled shell, then inspect the produced artifact tree with `check_artifacts --require-baseline-comparison`.
