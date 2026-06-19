# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 real-Chrome Colab Phase 0 artifact run completed and checked locally
- Changed: no code changes. Ran the real-Chrome CDP Colab bridge with shortcut run-all, captured rendered evidence, extracted 16 files from the notebook artifact bundle into `results/comparisons/colab_phase0`, and wrote `results/comparisons/colab_phase0/artifact_check.json`.
- Verified: `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot`; `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out results/comparisons/colab_phase0/artifact_check.json`. The checker reported status `pass`, 12/12 Phase 0 invariants passing, 9/9 artifact invariants passing, baseline comparison `pass`, baseline reference comparison `pass`, and accepted HEP alpha `0.25`.
- Blockers: none for Phase 0 infrastructure evidence. Colab artifacts and bridge evidence are present locally under ignored `results/` paths and are not tracked by git.
- Next step: write a concise Phase 0 report from the checked local baseline and checked Colab artifact tree, including the HEP alpha acceptance result and artifact-contract evidence, before adding pinned support or causal-coding mechanisms.
