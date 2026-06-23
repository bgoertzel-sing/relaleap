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
`results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep`,
and the broader seed-2 temporal comparison under
`results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2`,
and the broader seed-3 temporal comparison under
`results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3`,
and the broader seed-4 temporal comparison under
`results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4`,
the non-smoke temporal validation under
`results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep`,
the post-promotion validation PC-vs-supervised temporal-clipped comparison
under
`results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep`,
the objective-discriminative validation PC-vs-supervised temporal-clipped
comparison under
`results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate`,
the anchored-PC objective-gate validation comparison under
`results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate`,
the confidence-penalty objective-gate validation comparison under
`results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate`,
the margin-penalty objective-gate validation comparison under
`results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate`,
the label-smoothing objective-gate validation comparison under
`results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate`,
the focal objective-gate validation comparison under
`results/comparisons/colab_validation_focal_temporal_clipped_objective_gate`,
the extended focal objective-gate comparison under
`results/comparisons/colab_extended_focal_temporal_clipped_objective_gate`,
the larger focal objective-gate comparison under
`results/comparisons/colab_larger_focal_temporal_clipped_objective_gate`,
the tokenized larger focal objective-gate comparison under
`results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate`,
the xlarge char focal objective-gate comparison under
`results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate`,
the xxlarge char focal objective-gate comparison under
`results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate`,
the seed-2 tokenized larger focal-repeat objective-gate comparison under
`results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2`,
the seed-2 xxlarge char focal-repeat objective-gate comparison under
`results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2`,
the extended temporal support-stress check under
`results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep`,
the larger-char promotion-gate support-stress check under
`results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep`,
the non-char tokenized promotion-gate support-stress check under
`results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep`,
the residual capacity/support diagnostic validation under
`results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate`,
the larger-char/tokenized support-width validation under
`results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate`,
the seed-2 larger-char/tokenized support-width repeat under
`results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2`,
and the seed-3 larger-char/tokenized support-width promotion-gate evidence under
`results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3`.
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
by the current automation status. The default support-stress config uses
temporal-consistency clipped HEP; the pinned config remains the pinned-support
diagnostic path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress.yaml \
  --config configs/char_smoke_pinned_hep_support_stress.yaml \
  --out results/comparisons/colab_support_stress_pinned_vs_repicked
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_pinned_vs_repicked \
  --out results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check.json
```

The notebook also runs the clipped residual-adapter support-stress control
against the promoted temporal clipped support-stress default:

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

The notebook also runs the larger-char/tokenized support-width validation
comparison selected by the current automation status:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/artifact_check.json
```

The current focused Colab bridge target also runs the seed-2 support-width
repeat selected by `AUTOMATION_STATUS.md`:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2/artifact_check.json
```

The current promotion-gate bridge target also runs the seed-3 support-width
comparison selected by `AUTOMATION_STATUS.md`:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate_seed3.yaml \
  --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate_seed3.yaml \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3.yaml \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3 \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3/artifact_check.json
```

After extracting the seed-3 Colab artifact bundle locally, inspect the local and
Colab seed-3 artifact trees and write the support-width promotion-gate
satisfaction report without rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3 \
  --out results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed3/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report residual-support-width-promotion-gate-satisfaction \
  --out results/reports/residual_support_width_promotion_gate_satisfaction
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

The broader seed-2 label-free temporal check uses the same command-driven
harness and artifact checker:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed2.yaml \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2 \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/artifact_check.json
```

The broader seed-3 label-free temporal check uses the same command-driven
harness and artifact checker:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed3.yaml \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3 \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/artifact_check.json
```

The broader seed-4 label-free temporal check uses the same command-driven
harness and artifact checker:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed4.yaml \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4 \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/artifact_check.json
```

The non-smoke validation label-free temporal check uses the same command-driven
harness and artifact checker with sequence length `64`, hidden dimension `64`,
`12` residual columns, `3` HEP settling steps, and `25` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_support_stress_clipped.yaml \
  --config configs/char_validation_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_validation_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_validation_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
```

The post-promotion validation PC-vs-supervised temporal-clipped comparison uses
the same command-driven harness and artifact checker under the Colab-prefixed
validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_validation_pc_hep_support_stress_temporal_clipped.yaml \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/artifact_check.json
```

The objective-discriminative validation PC-vs-supervised temporal-clipped
comparison disables the support-stress preset while keeping the promoted
temporal clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/artifact_check.json
```

The anchored-PC objective-gate validation adds the local anchored PC objective
variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/artifact_check.json
```

The confidence-penalty objective-gate validation adds the first non-PC
residual-objective variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and extracted Colab confidence-penalty artifacts exist,
write the command-driven confidence-penalty objective decision report without
rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report confidence-penalty-residual-objective-decision \
  --out results/reports/confidence_penalty_residual_objective_decision
```

The margin-penalty objective-gate validation adds the next non-PC
residual-objective variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and extracted Colab margin-penalty artifacts exist, write
the command-driven margin-penalty objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report margin-penalty-residual-objective-decision \
  --out results/reports/margin_penalty_residual_objective_decision
```

The label-smoothing objective-gate validation adds the next non-PC
residual-objective variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_label_smoothing_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and extracted Colab label-smoothing artifacts exist, write
the command-driven label-smoothing objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report label-smoothing-residual-objective-decision \
  --out results/reports/label_smoothing_residual_objective_decision
```

The focal objective-gate validation adds the next non-PC residual-objective
variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_focal_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and extracted Colab focal artifacts exist, write the
command-driven focal objective decision report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report focal-residual-objective-decision \
  --out results/reports/focal_residual_objective_decision
```

The broader focal objective-gate check moves outside the current char
validation setting while preserving the objective-discriminative temporal
clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_extended_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_extended_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_extended_focal_temporal_clipped_objective_gate/artifact_check.json
```

The next larger-char focal objective-gate check reuses the temporal-clipped
promotion-gate scale while disabling the support-stress preset so learned
residual values are evaluated:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_larger_focal_temporal_clipped_objective_gate/artifact_check.json
```

The tokenized larger focal objective-gate check mirrors the non-char tokenized
promotion-gate scale while disabling the support-stress preset so learned
residual values are evaluated:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate/artifact_check.json
```

The seed-2 tokenized larger focal-repeat objective-gate check is the matching
Colab repeat for the local seed-2 tokenized focal evidence:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2/artifact_check.json
```

The xlarge char focal objective-gate check increases the char setting while
keeping the objective-discriminative temporal-clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/artifact_check.json
```

The xxlarge char focal objective-gate check is the next bounded scale step
after xlarge, with sequence length `192`, hidden dimension `160`, `40`
residual columns, and `70` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate/artifact_check.json
```

The seed-2 xxlarge char focal-repeat objective-gate check is the matching
Colab repeat for the local seed-2 xxlarge-char focal evidence:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2/artifact_check.json
```

The extended label-free temporal check moves outside the current char
validation setting with sequence length `96`, hidden dimension `64`, `16`
residual columns, `4` HEP settling steps, and `30` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_support_stress_clipped.yaml \
  --config configs/char_extended_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_extended_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_extended_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
```

The larger-char promotion-gate label-free temporal check uses sequence length
`128`, hidden dimension `96`, `24` residual columns, `4` HEP settling steps,
and `50` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_support_stress_clipped.yaml \
  --config configs/char_larger_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_larger_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_larger_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
```

The non-char tokenized promotion-gate label-free temporal check uses
deterministic word-token IDs with sequence length `64`, hidden dimension `96`,
`24` residual columns, `4` HEP settling steps, and `50` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_support_stress_clipped.yaml \
  --config configs/token_larger_hep_support_stress_entropy_clipped.yaml \
  --config configs/token_larger_hep_support_stress_temporal_clipped.yaml \
  --config configs/token_larger_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
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
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/metrics.csv
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/notes.md
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/artifact_check.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/runs/char_smoke_hep_support_stress_clipped_seed2/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/runs/char_smoke_hep_support_stress_entropy_clipped_seed2/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/runs/char_smoke_hep_support_stress_temporal_clipped_seed2/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed2/runs/char_smoke_hep_support_stress_guided_clipped_seed2/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/metrics.csv
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/notes.md
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/artifact_check.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/runs/char_smoke_hep_support_stress_clipped_seed3/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/runs/char_smoke_hep_support_stress_entropy_clipped_seed3/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/runs/char_smoke_hep_support_stress_temporal_clipped_seed3/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/runs/char_smoke_hep_support_stress_guided_clipped_seed3/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/metrics.csv
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/notes.md
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/artifact_check.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/runs/char_smoke_hep_support_stress_clipped_seed4/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/runs/char_smoke_hep_support_stress_entropy_clipped_seed4/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/runs/char_smoke_hep_support_stress_temporal_clipped_seed4/summary.json
results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/runs/char_smoke_hep_support_stress_guided_clipped_seed4/summary.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/summary.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/metrics.csv
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/notes.md
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_clipped/summary.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_entropy_clipped/summary.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_guided_clipped/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/metrics.csv
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/notes.md
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/artifact_check.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/runs/char_validation_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/runs/char_validation_pc_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/metrics.csv
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/notes.md
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/artifact_check.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/runs/char_validation_pc_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/metrics.csv
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/notes.md
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/artifact_check.json
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_pc_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/metrics.csv
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/notes.md
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/artifact_check.json
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/runs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate/summary.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/summary.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/metrics.csv
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/notes.md
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_clipped/summary.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_entropy_clipped/summary.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_guided_clipped/summary.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/summary.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/metrics.csv
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/notes.md
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_clipped/summary.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_entropy_clipped/summary.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_guided_clipped/summary.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/summary.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/metrics.csv
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/notes.md
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_clipped/summary.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_entropy_clipped/summary.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_temporal_clipped/summary.json
results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_guided_clipped/summary.json
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
