# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Phase 0 infrastructure: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in local comparison baseline, a local/Colab baseline gate, fail-closed live and after-the-fact artifact-contract inspection, child-run artifact-contract inspection, artifact-contract baseline gating, standard artifacts, and temporary Colab bridge support.

## Latest Run

- Status: ok, 2026-06-19 automation run at 2026-06-19T16:11:14Z
- Changed: tightened `relaleap.experiments.compare` so live comparison verdicts require each child run to report passing `summary_json`, `metrics_csv`, and `notes_md` artifact invariants; older artifact-unaware child summaries now fail during comparison instead of only during after-the-fact inspection. Added regression coverage for missing artifact-invariant reports and updated `README.md` plus `docs/colab_bridge.md`.
- Verified: `python -m unittest tests.test_compare`; `python -m compileall relaleap tests tools`; `python -m unittest tests.test_smoke tests.test_compare tests.test_check_artifacts` passed with 4 torch-dependent skips under bare Python; `.venv-conda/bin/python -m unittest tests.test_smoke tests.test_compare tests.test_check_artifacts`; `.venv-conda/bin/python -m relaleap.experiments.compare --out /tmp/relaleap-live-artifact-contract --baseline-reference baselines/phase0_char_smoke_comparison.json`; `.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir /tmp/relaleap-live-artifact-contract --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison --out /tmp/relaleap-live-artifact-contract/artifact_check_latest.json`; `git diff --check`. The checked comparison artifact tree passed with 12 Phase 0 model invariants, 9 artifact invariants, accepted HEP alpha `0.25`, child run artifact contracts present, and zero baseline mismatches.
- Blockers: live Colab/GPU execution was not attempted this run. Existing `.venv-conda` still emits the known NumPy 2.4.6 / torch 2.2.2 ABI warning on import; execution succeeds, and `pyproject.toml` already pins fresh installs to `numpy<2`.
- Next step: manually open the GitHub-backed Colab notebook, choose a GPU runtime, run all cells, then run `python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison`.
