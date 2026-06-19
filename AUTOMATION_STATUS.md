# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 bridge artifact-preservation hardening completed locally
- Changed: updated the Colab notebook to emit a rendered base64 zip bundle for `results/comparisons/colab_phase0`; updated `tools/colab_playwright_runner.py` to extract that bundle safely under the local repo root when present; documented the bundle behavior; added runner unit tests for safe extraction and path traversal rejection.
- Verified: `./.venv-conda/bin/python -m json.tool notebooks/relaleap_colab_smoke.ipynb >/dev/null`; `./.venv-conda/bin/python -m unittest tests.test_colab_playwright_runner`; `./.venv-conda/bin/python -m unittest tests.test_check_artifacts`. `pytest` is not installed in `.venv-conda`, so the focused tests were run with `unittest`.
- Blockers: no local code blocker. The current checked-out repo still lacks a local `results/comparisons/colab_phase0` Colab artifact tree; the previous rendered evidence cannot reconstruct child `metrics.csv` and `notes.md` files by itself. The bundle extraction path will take effect after this notebook/runner update is available to the GitHub-backed Colab run.
- Next step: run `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot`, retry once with `--run-method both` if the shortcut fails, then run `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out results/comparisons/colab_phase0/artifact_check.json`.
