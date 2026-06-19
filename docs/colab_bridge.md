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

Then open Colab and log in once:

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

Expected artifacts:

```text
results/comparisons/colab_phase0/baseline_comparison.json
results/comparisons/colab_phase0/metrics.csv
results/comparisons/colab_phase0/notes.md
results/comparisons/colab_phase0/summary.json
results/comparisons/colab_phase0/runs/char_smoke/summary.json
results/comparisons/colab_phase0/runs/char_smoke_pc/summary.json
results/comparisons/colab_phase0/runs/char_smoke_hep/summary.json
```

The checked-in local baseline currently accepts HEP alpha `0.25`. The
`--baseline-reference` gate writes `baseline_comparison.json` and exits nonzero
if the Colab/GPU run changes the accepted HEP alpha, loses Phase 0 invariants,
loses required artifact invariants, or changes the comparison config set.

After a manual Colab run, inspect the artifact directory without rerunning the
experiments:

```bash
python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

The checker fails if required comparison or per-run artifacts are missing, if
the comparison verdict is not `pass`, if the completed summary reports failed
or missing artifact invariants, or if the completed comparison summary drifts
from the checked-in Phase 0 baseline. Add
`--require-baseline-comparison` when the run was expected to write
`baseline_comparison.json` during execution.
