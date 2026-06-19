# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Post-Phase 0 pinned-support evidence review: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, and a GitHub-backed Colab notebook/bridge contract that now has successful visible pinned-support evidence.

## Latest Run

- Status: ok, 2026-06-19 bounded real-Chrome Colab bridge pinned-evidence run completed
- Changed: ran the GitHub-backed Colab notebook through Chrome CDP with `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot`; no code changes were needed. The bridge wrote rendered evidence to `results/colab_bridge_evidence/latest_colab_output.txt` and extracted 20 artifact files into `results/comparisons/colab_phase0` and `results/runs/colab_char_smoke_pinned_hep`.
- Verified: the bridge evidence passed its visible-output marker checks, including `cuda_available: True`, default comparison `"status": "pass"`, accepted HEP alpha output, `"pinned_support": true`, `Pinned HEP status: ok`, and `RelaLeap Colab Phase 0 comparison completed.`; `./.venv-conda/bin/python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json --require-baseline-comparison` returned `status: pass`, 12/12 Phase 0 invariants passing, 9/9 artifact invariants passing, baseline comparison/reference status `pass`, and accepted HEP alpha `0.25`; `results/runs/colab_char_smoke_pinned_hep/summary.json` reports status `ok`, experiment `char_smoke_pinned_hep`, `phase0.pinned_support: true`, 4/4 Phase 0 invariants passing, required artifacts passing, and alpha-0 logit delta `0.0`.
- Blockers: none for the Colab pinned-support evidence handoff. Generated `results/` artifacts remain ignored by git, so the tracked record of this run is this status file plus the local artifact tree.
- Next step: compare the pinned-support Colab HEP sweep against the ordinary `char_smoke_hep` sweep and decide whether pinned support needs a separate checked baseline or should remain an opt-in artifact-only smoke path.
