# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 verified real-Chrome CLI Colab/GPU bridge
- Changed: hardened `tools/colab_playwright_runner.py` so `--wait-completion` keeps watching for Colab run-confirmation modals and clicks `Run anyway`/related buttons during the whole completion wait, rather than only once immediately after `Run all`.
- Verified: Ben reran the real-Chrome CDP Colab bridge from CLI and reported successful completion with accepted HEP alpha `0.25`, loss improvement `0.011214256286621094`, and `RelaLeap Colab Phase 0 comparison completed.` `.venv-conda/bin/python -m py_compile tools/colab_playwright_runner.py` and `git diff --check` passed after the bridge hardening.
- Blockers: no current RelaLeap research blocker for CLI-launched Colab/GPU execution when Chrome is running with `--remote-debugging-port=9222`, logged into Google/Colab, and a GPU runtime is available. The Codex automation sandbox itself still cannot probe that localhost CDP port, so live GPU bridge launches should continue from the CLI/shell context for now.
- Next step: inspect or preserve the Colab evidence/artifact outputs from the successful CLI run, then continue Phase 0 experiments with the real-Chrome CLI bridge as the GPU path.
