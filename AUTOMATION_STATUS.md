# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 HEP next-step selection: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, successful visible pinned-support Colab evidence, local paired ordinary-vs-pinned HEP comparison, support-instability diagnostics that expose whether ordinary repicked settling changes column support and whether pinned-vs-repicked logits diverge, an opt-in HEP support-stress config that intentionally produces nonzero support repicking without changing the checked Phase 0 baseline, a paired pinned-support support-stress comparison path with run-level support diagnostics in comparison artifacts, and Colab notebook/bridge validation prepared for the paired support-stress artifact run.

## Latest Run

- Status: blocked, 2026-06-19 bounded Colab support-stress bridge-prep increment completed; real-Chrome CDP run intentionally not started because the GitHub-backed Colab notebook will not include this local notebook change until this commit is pushed.
- Changed: extended `notebooks/relaleap_colab_smoke.ipynb` to run `configs/char_smoke_hep_support_stress.yaml` versus `configs/char_smoke_pinned_hep_support_stress.yaml`, write `results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check.json`, assert and print support-stress status markers, and include that comparison in the rendered artifact bundle; tightened `tools/colab_playwright_runner.py` visible-output validation so the bridge cannot pass on the older Phase 0/pinned-smoke-only notebook output; updated bridge docs and focused tests.
- Verified: `./.venv-conda/bin/python -m json.tool notebooks/relaleap_colab_smoke.ipynb` passed; `./.venv-conda/bin/python -m unittest tests.test_colab_playwright_runner` passed 4 tests; `./.venv-conda/bin/python -m unittest discover -s tests` passed 38 tests.
- Blockers: the notebook URL opened by the real-Chrome CDP bridge is GitHub-backed, so the support-stress Colab run depends on this local commit being pushed before executing the requested bridge command. Running the bridge before that push would duplicate completed Phase 0/pinned-smoke evidence and fail the new support-stress evidence gate.
- Next step: after this commit is pushed, run `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot`.
