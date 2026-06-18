# Colab Bridge

This is a temporary Colab execution bridge for RelaLeap GPU smoke tests.

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

Install Playwright locally:

```bash
python -m pip install playwright
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

The smoke notebook runs:

```bash
python -m relaleap.experiments.run --config configs/char_smoke.yaml --out results/runs/colab_smoke
```

Expected artifacts:

```text
results/runs/colab_smoke/config.yaml
results/runs/colab_smoke/metrics.csv
results/runs/colab_smoke/notes.md
results/runs/colab_smoke/summary.json
```

