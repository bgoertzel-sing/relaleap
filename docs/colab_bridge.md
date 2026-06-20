# Colab Bridge

This is a temporary Colab execution bridge for RelaLeap GPU Phase 0 smoke tests.

Colab is not the source of truth. The GitHub repo, config files, and run artifacts are the source of truth. The notebook should only clone/pull the repo, install it, run a config-driven command, and show/save the artifacts.

## Notebook URL

Open:

```text
https://colab.research.google.com/github/bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb
```

In Colab, choose a GPU runtime before running:

```text
Runtime -> Change runtime type -> GPU
```

## Browser Automation Hack

Do not use an old conda `base` environment for this. The bridge needs a modern
Python, and the RelaLeap package itself declares Python 3.10 or newer.

Fast path:

```bash
bash tools/setup_colab_bridge.sh
```

Manual path:

```bash
CONDA_PKGS_DIRS=$PWD/.conda-pkgs conda create -p $PWD/.venv-conda python=3.11 -y
conda activate $PWD/.venv-conda
python -m pip install --upgrade pip
python -m pip install -e '.[colab-bridge]' --no-build-isolation
python -m playwright install chromium
```

### Preferred real-Chrome route

Google may block login from Playwright's bundled testing browser. The more
usable route is to launch ordinary Chrome once with remote debugging enabled,
log into Google/Colab there, and let the helper attach to that logged-in
session.

Launch Chrome:

```bash
open -na "Google Chrome" --args \
  --remote-debugging-port=9222 \
  --user-data-dir=/Users/bengoertzel/.codex-relaleap-chrome
```

Then run the notebook through that Chrome session:

```bash
./.venv-conda/bin/python tools/colab_playwright_runner.py \
  --cdp-url http://127.0.0.1:9222 \
  --run-all \
  --run-method shortcut \
  --wait-completion \
  --debug-snapshot
```

The helper clicks the current Colab "Run anyway" trust modal when it appears,
keeps watching for common resume/reconnect prompts while it waits, waits for
`RelaLeap Colab Phase 0 comparison completed.`, and saves visible notebook
evidence to:

```text
results/colab_bridge_evidence/latest_colab_output.txt
```

The notebook also emits a rendered base64 zip bundle for
`results/comparisons/colab_phase0`, the pinned-support smoke run under
`results/runs/colab_char_smoke_pinned_hep`, and the paired support-stress
comparison under
`results/comparisons/colab_support_stress_pinned_vs_repicked`, the clipped HEP
support-stress comparison under
`results/comparisons/colab_support_stress_clipped_hep`, and the guided clipped
HEP support-stress comparison under
`results/comparisons/colab_support_stress_guided_clipped_hep`, and the
temporal-vs-entropy guided clipped support-stress comparison under
`results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep`.
When that bundle is present in the rendered output, the helper extracts it
under the local repo root so the normal after-the-fact checker can inspect the
Colab artifact tree locally.

If the keyboard shortcut stops working after a Colab UI change, try
`--run-method both`.

### Bundled-browser fallback

Open Colab and log in once:

```bash
python tools/colab_playwright_runner.py --manual-login
```

After the profile is logged in, try a best-effort run-all:

```bash
python tools/colab_playwright_runner.py --run-all
```

This is brittle by design. It may fail when Google changes the Colab UI, requires additional auth, shows a quota warning, or disconnects the runtime.

## Expected Artifacts

The smoke notebook runs the same command-driven comparison used locally:

```bash
python -m relaleap.experiments.compare --out results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

It also runs the opt-in pinned-support HEP smoke config as a separate
command-driven artifact check:

```bash
python -m relaleap.experiments.run --config configs/char_smoke_pinned_hep.yaml --out results/runs/colab_char_smoke_pinned_hep
```

The notebook also runs the support-instability diagnostic comparison selected
by the current automation status:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress.yaml \
  --config configs/char_smoke_pinned_hep_support_stress.yaml \
  --out results/comparisons/colab_support_stress_pinned_vs_repicked
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_pinned_vs_repicked \
  --out results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check.json
```

The notebook also runs the opt-in clipped HEP support-stress comparison:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress.yaml \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --out results/comparisons/colab_support_stress_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_clipped_hep \
  --out results/comparisons/colab_support_stress_clipped_hep/artifact_check.json
```

The notebook also runs the guided clipped HEP support-stress comparison:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_support_stress_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_guided_clipped_hep \
  --out results/comparisons/colab_support_stress_guided_clipped_hep/artifact_check.json
```

After extracting a completed guided clipped artifact bundle locally, write the
oracle decision report without rerunning Colab:

```bash
python -m relaleap.experiments.decision_report \
  --report guided-clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_guided_clipped_hep \
  --artifact-check results/comparisons/colab_support_stress_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/guided_clipped_hep_decision
```

The deployable label-free entropy-gradient clipped HEP probe can be run against
the clipped baseline and guided oracle with the same command-driven harness:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_support_stress_entropy_vs_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_entropy_vs_guided_clipped_hep \
  --out results/comparisons/colab_support_stress_entropy_vs_guided_clipped_hep/artifact_check.json
```

The deployable label-free temporal-consistency clipped HEP probe can be run
against the clipped baseline, entropy probe, and guided oracle with the same
command-driven harness:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
```

After extracting the completed temporal comparison artifact bundle locally,
write the label-free candidate decision report without rerunning Colab:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_decision
```

Expected artifacts:

```text
results/comparisons/colab_phase0/baseline_comparison.json
results/comparisons/colab_phase0/metrics.csv
results/comparisons/colab_phase0/notes.md
results/comparisons/colab_phase0/summary.json
results/comparisons/colab_phase0/runs/char_smoke/summary.json
results/comparisons/colab_phase0/runs/char_smoke_pc/summary.json
results/comparisons/colab_phase0/runs/char_smoke_hep/summary.json
results/runs/colab_char_smoke_pinned_hep/summary.json
results/runs/colab_char_smoke_pinned_hep/metrics.csv
results/runs/colab_char_smoke_pinned_hep/notes.md
results/comparisons/colab_support_stress_pinned_vs_repicked/summary.json
results/comparisons/colab_support_stress_pinned_vs_repicked/metrics.csv
results/comparisons/colab_support_stress_pinned_vs_repicked/notes.md
results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check.json
results/comparisons/colab_support_stress_pinned_vs_repicked/runs/char_smoke_hep_support_stress/summary.json
results/comparisons/colab_support_stress_pinned_vs_repicked/runs/char_smoke_pinned_hep_support_stress/summary.json
results/comparisons/colab_support_stress_clipped_hep/summary.json
results/comparisons/colab_support_stress_clipped_hep/metrics.csv
results/comparisons/colab_support_stress_clipped_hep/notes.md
results/comparisons/colab_support_stress_clipped_hep/artifact_check.json
results/comparisons/colab_support_stress_clipped_hep/runs/char_smoke_hep_support_stress/summary.json
results/comparisons/colab_support_stress_clipped_hep/runs/char_smoke_hep_support_stress_clipped/summary.json
results/comparisons/colab_support_stress_guided_clipped_hep/summary.json
results/comparisons/colab_support_stress_guided_clipped_hep/metrics.csv
results/comparisons/colab_support_stress_guided_clipped_hep/notes.md
results/comparisons/colab_support_stress_guided_clipped_hep/artifact_check.json
results/comparisons/colab_support_stress_guided_clipped_hep/runs/char_smoke_hep_support_stress_clipped/summary.json
results/comparisons/colab_support_stress_guided_clipped_hep/runs/char_smoke_hep_support_stress_guided_clipped/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/metrics.csv
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/notes.md
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_smoke_hep_support_stress_clipped/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_smoke_hep_support_stress_entropy_clipped/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_smoke_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_smoke_hep_support_stress_guided_clipped/summary.json
```

The checked-in schema v3 local baseline currently accepts HEP alpha `0.25`. The
`--baseline-reference` gate writes `baseline_comparison.json` and exits nonzero
if the Colab/GPU run changes the accepted HEP alpha, loses Phase 0 invariants,
loses required aggregate or per-run artifact invariants, omits the child-run
artifact-invariant contract, or changes the comparison config set.

After a manual Colab run, inspect the artifact directory without rerunning the
experiments:

```bash
python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

The checker fails if required comparison or per-run artifacts are missing, if
the comparison verdict is not `pass`, if the completed summary reports failed
or missing artifact invariants, if a child run summary belongs to the wrong
experiment, omits, or fails its own artifact-invariant contract, or if the
completed comparison summary drifts from the checked-in Phase 0 baseline. Add
`--require-baseline-comparison` when the run was expected to write
`baseline_comparison.json` during execution.
