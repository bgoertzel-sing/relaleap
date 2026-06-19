# RelaLeap Automation Status

This file is maintained by the RelaLeap research automation so progress can be checked from GitHub, including from a phone.

## Current Focus

Pinned-support Colab handoff after Phase 0: command-driven experiment harness, char-level smoke invariants, configurable supervised and PC-style residual smoke training, compact objective and HEP alpha reports with acceptance policy, a checked-in schema v3 local comparison baseline, local/Colab baseline gates, fail-closed live and after-the-fact artifact-contract inspection, per-run child artifact and identity inspection, artifact-contract baseline gating, standard artifacts, temporary real-Chrome Colab bridge support, a concise Phase 0 report, an opt-in pinned-support HEP smoke path, and a GitHub-backed Colab notebook/bridge contract that now requires visible pinned-support evidence.

## Latest Run

- Status: ok, 2026-06-19 bounded Colab bridge pinned-evidence prep run completed
- Changed: updated `notebooks/relaleap_colab_smoke.ipynb` so the GitHub-backed Colab run still performs the default Phase 0 baseline comparison and now also runs `configs/char_smoke_pinned_hep.yaml` into `results/runs/colab_char_smoke_pinned_hep`; the rendered artifact bundle now includes both artifact trees. Tightened `tools/colab_playwright_runner.py` so completion evidence must include visible pinned-support summary output and `Pinned HEP status: ok`; updated the bridge tests and documentation accordingly. The default Phase 0 comparison config set and checked schema v3 baseline remain unchanged.
- Verified: `./.venv-conda/bin/python -m unittest discover -s tests` ran 36 tests passing; `./.venv-conda/bin/python -m relaleap.experiments.run --config configs/char_smoke_pinned_hep.yaml --out results/runs/colab_char_smoke_pinned_hep_local_check` completed locally with status `ok`, `phase0.pinned_support: true`, 4/4 Phase 0 invariants passing, required artifacts passing, and HEP alpha-0 delta `0.0`; the notebook JSON parses and contains the pinned-support command; Chrome CDP at `http://127.0.0.1:9222/json/version` responded with Chrome `150.0.7871.24`.
- Blockers: the real-Chrome Colab bridge opens the GitHub-backed notebook URL, so a meaningful pinned-support Colab run requires this notebook/bridge commit to be pushed first. Running the bridge before GitHub has the updated notebook would duplicate the old default-only Colab path and fail the new pinned-evidence marker gate.
- Next step: after the auto-push helper publishes this commit, run `./.venv-conda/bin/python tools/colab_playwright_runner.py --cdp-url http://127.0.0.1:9222 --run-all --run-method shortcut --wait-completion --debug-snapshot` and preserve the extracted default comparison plus pinned-support artifact evidence.
