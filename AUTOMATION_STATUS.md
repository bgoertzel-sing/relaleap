# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, a local/Colab baseline gate, fail-closed live and after-the-fact artifact-contract inspection, child-run artifact and identity inspection, artifact-contract baseline gating, standard artifacts, and temporary real-Chrome Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 child-run identity contract hardening
- Changed: tightened `relaleap.experiments.check_artifacts` so each child run `summary.json` must report the same `experiment_id` as the comparison entry that points to it. Added regression coverage and updated the README/Colab bridge docs to describe stale or swapped child-run evidence as a fail-closed condition.
- Verified: `python -m unittest tests.test_check_artifacts tests.test_compare tests.test_smoke` passed with 4 torch skips; `python -m compileall relaleap tests tools` passed; `.venv-conda/bin/python -m unittest tests.test_check_artifacts tests.test_compare tests.test_smoke` passed; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-child-identity-contract --baseline-reference baselines/phase0_char_smoke_comparison.json` passed with accepted HEP alpha `0.25`; `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir /tmp/relaleap-child-identity-contract --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out /tmp/relaleap-child-identity-contract/artifact_check_latest.json` passed with zero failures.
- Blockers: live Colab/GPU execution was not attempted in this local-first hardening run. No current RelaLeap research blocker if Chrome is running with `--remote-debugging-port=9222`, logged into Google/Colab, and a GPU runtime is available; Colab auth, quota, disconnects, or runtime popups can still require intervention.
- Next step: run the real-Chrome CDP Colab bridge once more with `--wait-completion --debug-snapshot`, then inspect the produced artifact tree with `check_artifacts --require-baseline-comparison`.
