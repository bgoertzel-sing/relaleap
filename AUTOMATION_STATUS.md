# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, a local/Colab baseline gate, fail-closed live and after-the-fact artifact-contract inspection, child-run artifact-contract inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 bridge update after Ben's successful live Colab/GPU smoke run
- Changed: upgraded `tools/colab_playwright_runner.py` so the real-Chrome CDP bridge can run unattended after launch, click the current Colab trust modal, wait for the final completion text, save visible notebook evidence, and fail clearly on timeout or missing pass/HEP output. Updated `docs/colab_bridge.md` with the preferred real-Chrome route.
- Verified: Ben reported the Chrome/CDP Colab notebook run completed and printed accepted HEP alpha `0.25`, loss improvement `0.011214256286621094`, and `RelaLeap Colab Phase 0 comparison completed.` Local bridge syntax and docs were updated after that live validation.
- Blockers: no current RelaLeap research blocker if Chrome is running with `--remote-debugging-port=9222`, logged into Google/Colab, and a GPU runtime is available. Colab auth, quota, disconnects, or runtime popups can still require intervention.
- Next step: have the hourly automation use the real-Chrome CDP bridge with `--wait-completion` when GPU execution is needed, while continuing to keep local command-driven tests as the source of truth.
