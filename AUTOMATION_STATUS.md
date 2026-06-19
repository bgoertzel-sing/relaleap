# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 local Phase 0 baseline/artifact gate reverified while continuing the Colab evidence hardening loop
- Changed: hardened `tools/colab_playwright_runner.py` so `--wait-completion` waits for the completion marker inside rendered Colab output elements, writes a rendered-output evidence section before full page text, and rejects source-only notebook captures that lack `cuda_available: True`, `"status": "pass"`, accepted HEP alpha output, and `RelaLeap Colab Phase 0 comparison completed.` Also made `check_artifacts` tolerate a malformed non-object comparison verdict without crashing.
- Verified: `.venv-conda/bin/python -m unittest discover -s tests` passed. Local source-of-truth run passed with `.venv-conda/bin/python -m relaleap.experiments.compare --out results/comparisons/local_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json`, accepting HEP alpha `0.25` with loss improvement `0.011214256286621094`. After-the-fact artifact inspection passed with `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/local_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out results/comparisons/local_phase0/artifact_check.json`.
- Blockers: no current local Phase 0 blocker. The existing `results/colab_bridge_evidence/latest_colab_output.txt` is only notebook/page text from before the rendered-output evidence hardening, so it should not be treated as preserved GPU evidence. Live GPU bridge launches still need the real-Chrome CLI context with Chrome running on `--remote-debugging-port=9222`, logged into Google/Colab, and a GPU runtime available.
- Next step: rerun the real-Chrome CLI Colab bridge with the hardened evidence capture, then inspect `results/colab_bridge_evidence/latest_colab_output.txt` for rendered GPU/pass/accepted-alpha evidence before continuing larger Phase 0 experiments.
