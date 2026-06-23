# RelaLeap

RelaLeap is an experimental codebase for residual layer learning prototypes, beginning with columnar PC/CC residual layers on frozen backpropagation-trained transformer bases.

The first goal is not frontier-scale performance. The first goal is a reproducible small-scale methodology for testing:

- frozen-base residual adaptation;
- sparse columnar rank-one residual atoms;
- predictive-coding residual training;
- highway error propagation;
- windowed settling and pinned support;
- later causal-coding audits and simple symbolic heads.

## Current Scientific Pivot

As of 2026-06-23, the research loop should treat CE/perplexity as guardrails
rather than the main success signal. The central question is now whether
residual columns learn causally separable, reusable corrections with less
interference than matched dense or sparse alternatives. The next experiments
should prioritize support-width deconfounding, oracle-support regret,
functional churn, causal intervention fingerprints, task-free continual
learning, finite-update commutators, and dense-teacher residual distillation.
See `docs/research_pivot_2026_06_23.md`.

## Initial Mission

Validate a minimal experimental harness on char-level Tiny Shakespeare before moving to larger GPU experiments.

## First Campaign

Run only infrastructure and early residual/HEP experiments first:

1. Phase 0 invariants.
2. Tiny BP base: 2 layers, hidden dim 32, char vocabulary, sequence length 32.
3. Freeze base.
4. One-site columns: 8 columns, 4 atoms per column, top-k 1.
5. PC residual training.
6. HEP alpha sweep at small inference-step counts.
7. Report before adding pinned support, causal-coding mechanisms, or symbolic heads.

## Early Guardrails

- Zero-initialized columns must preserve base logits within tolerance.
- Frozen base parameters must not change during residual-layer training.
- HEP with alpha 0 must match ordinary inference.
- Every run must produce `summary.json`, `metrics.csv`, and `notes.md`.
- Stop after three infrastructure failures or any invariant failure.

## Execution Model

Every experiment should be runnable as a plain Python command so it can run locally, in Colab, or on another GPU backend:

```bash
python -m relaleap.experiments.run --config configs/char_smoke.yaml
```

The current supervised/PC/HEP char smoke comparison is also command-driven:

```bash
python -m relaleap.experiments.compare
```

The comparison `summary.json` includes a compact verdict with aggregate
Phase 0 invariant pass/fail status, required artifact pass/fail status, the
best HEP alpha by loss, and an HEP alpha acceptance policy. By default,
accepted nonzero HEP alphas must improve loss over alpha 0 while keeping
ordinary-logit delta at or below `0.1`; tune that gate with
`--hep-max-logit-delta` and `--hep-min-loss-improvement`.
To refresh the checked-in compact Phase 0 baseline after an intentional
methodology change, run:

```bash
python -m relaleap.experiments.compare --baseline-out baselines/phase0_char_smoke_comparison.json
```

The current schema v3 baseline records all 12 Phase 0 model invariants and all
9 child-run artifact invariants passing, pins each child run's artifact
contract status, and accepts HEP alpha `0.25` under the default logit-delta
policy.

To compare a fresh local or Colab/GPU run against that baseline, run:

```bash
python -m relaleap.experiments.compare --out results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

This writes `baseline_comparison.json` under the comparison output directory
and exits nonzero if the accepted HEP alpha, Phase 0 invariant result, aggregate
or per-run artifact contract, or config set diverges from the checked-in
baseline. The comparison verdict fails closed when a child run summary does not
report passing `summary_json`, `metrics_csv`, and `notes_md` artifact
invariants.

To inspect a completed local or Colab comparison artifact tree without rerunning
experiments, run:

```bash
python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/colab_phase0 --baseline-reference baselines/phase0_char_smoke_comparison.json
```

This checks the required `summary.json`, `metrics.csv`, and `notes.md` artifacts
for the comparison and child runs, then reports the verdict, Phase 0 invariant
status, artifact invariant status, accepted HEP alpha, and whether the completed
summary still matches the checked-in local baseline. Add
`--require-baseline-comparison` when the run was expected to write
`baseline_comparison.json` during execution. The checker fails closed when a
completed comparison summary is missing the artifact-invariant verdict fields,
or when a child run `summary.json` belongs to the wrong experiment, omits, or
fails its own artifact-invariant contract, which catches stale, swapped, or
older artifact-unaware summaries before they are treated as valid Colab/GPU
evidence.

The current tiny HEP alpha sweep is also command-driven:

```bash
python -m relaleap.experiments.run --config configs/char_smoke_hep.yaml --out results/runs/char_smoke_hep
```

Pinned-support HEP settling has a separate opt-in smoke config. It reuses the
ordinary-pass top-k residual-column support during later settling updates, while
leaving the default Phase 0 comparison baseline unchanged:

```bash
python -m relaleap.experiments.run --config configs/char_smoke_pinned_hep.yaml --out results/runs/char_smoke_pinned_hep
```

To compare pinned-support HEP against the ordinary HEP smoke path without
changing the checked Phase 0 baseline, run:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep.yaml \
  --config configs/char_smoke_pinned_hep.yaml \
  --out results/comparisons/pinned_vs_ordinary_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/pinned_vs_ordinary_hep
```

The current smoke evidence shows identical ordinary and pinned HEP alpha sweeps,
with `support_change_fraction` and `pinned_vs_repicked_logit_delta` both `0.0`
for every alpha. That means this smoke case does not yet exercise support
repicking during settling, so pinned support remains an opt-in artifact-only
smoke path rather than a separate checked baseline.

The default support-stress config intentionally reshapes the trained residual
columns after the ordinary smoke update so the support-instability diagnostic
sees nonzero repicking without changing the checked Phase 0 baseline. After the
promotion-gate evidence passed, this support-stress path now uses the
deployable temporal-consistency settling signal with the `0.01` per-token clip:

```bash
python -m relaleap.experiments.run \
  --config configs/char_smoke_hep_support_stress.yaml \
  --out results/runs/char_smoke_hep_support_stress
```

This is the default support-stress mitigation path. It remains outside the
checked Phase 0 comparison baseline.

A separate clipped residual-adapter support-stress control bounds each settling
update by per-token hidden-state norm while leaving the settling objective on
the residual adapter. This preserves the earlier control path for comparison
against the promoted temporal clipped default:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress.yaml \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --out results/comparisons/support_stress_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_clipped_hep
```

After a completed local or Colab clipped support-stress comparison, write the
command-driven clipped-HEP decision report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_clipped_hep \
  --artifact-check results/comparisons/colab_support_stress_clipped_hep/artifact_check_local.json \
  --out results/reports/clipped_hep_decision
```

The report writes `decision_report.json` and `decision_report.md`. The clipped
residual-adapter control remains useful as comparison evidence, but the default
support-stress mitigation path is temporal clipped HEP after the promotion gate.

An additional opt-in guided clipped HEP support-stress config uses a supervised
cross-entropy hidden-state gradient step during settling. This is a diagnostic
oracle probe, not a deployable inference path, and asks whether the clipped
settling budget can support any loss-improving error step under support stress:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/support_stress_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_guided_clipped_hep
```

The local smoke evidence accepts the guided clipped alpha `1.0`, improving loss
over alpha 0 while staying within the default ordinary-logit delta budget.
After a completed local or Colab guided clipped comparison, write the
command-driven guided-clipped oracle decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report guided-clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_guided_clipped_hep \
  --artifact-check results/comparisons/colab_support_stress_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/guided_clipped_hep_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current default policy, guided clipped HEP can confirm that a supervised
gradient oracle improves support-stress loss within the stability budgets, but
it remains diagnostic-only and cannot be promoted until a deployable inference
error signal is selected.

An opt-in label-free clipped HEP support-stress config uses prediction-entropy
hidden-state gradient descent during settling. This is deployable at inference
time because it uses model logits rather than supervised labels:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/support_stress_entropy_vs_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_entropy_vs_guided_clipped_hep
```

This keeps the checked Phase 0 baseline unchanged while comparing a deployable
label-free signal with the unguided clipped baseline and the supervised guided
oracle on the same support-stress path.

An additional opt-in label-free clipped HEP support-stress config uses temporal
next-token consistency during settling. It distills each position's detached
current prediction into the following position's prediction, so it can run
without labels while probing a different deployable error signal than entropy
minimization:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep
```

This keeps temporal consistency diagnostic-only until command-driven local and
Colab evidence show a clipped nonzero temporal alpha improves support-stress
loss within the ordinary-logit budget.
After a completed local or Colab temporal comparison, write the command-driven
temporal-clipped HEP decision report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current default policy, temporal clipped HEP is selected as the deployable
label-free support-stress mitigation candidate only when a nonzero temporal
alpha improves loss while staying inside the ordinary-logit and
pinned-vs-repicked delta budgets; default promotion still requires broader
evidence than the current smoke comparison.

A broader opt-in temporal clipped HEP support-stress comparison repeats the
same clipped baseline, entropy, temporal, and guided oracle probe at seed 2.
This is the next label-free candidate check before considering any default
support-stress mitigation:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed2.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed2.yaml \
  --out results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed2
```

This keeps the checked Phase 0 baseline unchanged while testing whether the
seed-1 temporal consistency signal survives a seed change.

The next broader local temporal clipped HEP support-stress comparison repeats
the same candidate check at seed 3 before any default support-stress promotion:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed3.yaml \
  --out results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed3
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed3
```

This keeps temporal consistency selected only as the current label-free
candidate while checking whether the seed-1 and seed-2 pattern survives another
deterministic seed.

The matching GitHub-backed Colab seed-3 validation uses the same command-driven
harness and writes the artifact tree under the Colab-prefixed comparison path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed3.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed3.yaml \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3
```

After a completed Colab run and local artifact extraction, inspect the
extracted artifact tree and write the seed-3 temporal decision report without
rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3 \
  --out results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3 \
  --artifact-check results/comparisons/colab_support_stress_temporal_vs_entropy_guided_clipped_hep_seed3/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_seed3_colab_decision
```

The next broader local temporal clipped HEP support-stress comparison repeats
the same candidate check at seed 4:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_entropy_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_temporal_clipped_seed4.yaml \
  --config configs/char_smoke_hep_support_stress_guided_clipped_seed4.yaml \
  --out results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed4
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed4 \
  --out results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed4 \
  --artifact-check results/comparisons/support_stress_temporal_vs_entropy_guided_clipped_hep_seed4/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_seed4_local_decision
```

The seed-4 local evidence again selects temporal consistency as the deployable
label-free candidate while keeping default promotion blocked pending broader
evidence.

The matching GitHub-backed Colab seed-4 validation uses the same command-driven
harness and writes the artifact tree under the Colab-prefixed comparison path:

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

After seed-1 through seed-4 local and Colab temporal decision reports exist,
write the command-driven aggregate report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep-aggregate \
  --out results/reports/temporal_clipped_hep_multiseed_aggregate
```

The aggregate fails closed if any expected decision report is missing, fails,
does not select temporal consistency, or lacks an accepted nonzero temporal
alpha inside the ordinary-logit and pinned-vs-repicked delta budgets. It can
select temporal consistency as the current label-free support-stress candidate
across smoke seed evidence, but still blocks default promotion pending broader
non-smoke validation.

A broader opt-in non-smoke temporal clipped HEP validation repeats the clipped
baseline, entropy, temporal, and guided oracle probe with sequence length `64`,
hidden dimension `64`, `12` residual columns, `3` HEP settling steps, and `25`
training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_support_stress_clipped.yaml \
  --config configs/char_validation_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_validation_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_validation_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/validation_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/validation_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_validation_local_decision
```

This keeps the checked Phase 0 baseline unchanged while testing whether the
temporal label-free signal survives a larger deterministic char validation
setting before any default-promotion decision.

After temporal clipped HEP is promoted as the default support-stress mitigation,
the next residual-layer learning check compares supervised residual training
against PC-style logit-MSE residual training under the same temporal clipped
validation setting:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_validation_pc_hep_support_stress_temporal_clipped.yaml \
  --out results/comparisons/validation_pc_vs_supervised_temporal_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_pc_vs_supervised_temporal_clipped_hep \
  --out results/comparisons/validation_pc_vs_supervised_temporal_clipped_hep/artifact_check_local.json
```

This keeps the checked Phase 0 baseline unchanged while returning the
post-promotion research loop to supervised-vs-PC residual objective evidence
under the promoted deployable temporal clipped HEP path.

The support-stress validation pair above intentionally preserves historical
support-repicking evidence by rewriting residual atom values before the HEP
sweep, so it is not objective-discriminative after residual training. The
objective-discriminative local gate keeps the same promoted temporal clipped
HEP inference path but disables that support-stress preset, so CE/HEP behavior
is measured on the learned supervised or PC residual values:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate \
  --out results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching GitHub-backed PC validation uses the same command-driven harness
and writes the artifact tree under its Colab-prefixed validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_validation_pc_hep_support_stress_temporal_clipped.yaml \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_hep/artifact_check.json
```

The matching GitHub-backed objective-discriminative PC validation disables the
support-stress preset while preserving the promoted temporal clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and Colab objective-gate artifacts exist, write the
command-driven supervised-vs-PC residual-objective decision report without
rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-objective-gate \
  --out results/reports/residual_objective_gate_decision
```

The report writes `decision_report.json` and `decision_report.md`. The current
artifact-backed local and Colab objective-gate evidence keeps supervised CE as
the default residual objective: both supervised and PC runs improve their own
training losses with the support-stress preset disabled, but PC has worse
supervised CE HEP loss than supervised residual training in both backends.

To inspect that PC objective gap without rerunning local or Colab experiments,
write the command-driven diagnostic report over the same artifact-backed
objective-gate evidence:

```bash
python -m relaleap.experiments.decision_report \
  --report pc-residual-objective-diagnostics \
  --out results/reports/pc_residual_objective_diagnostics
```

The report writes `decision_report.json` and `decision_report.md`. It is
diagnostic-only: it quantifies the PC-minus-supervised best HEP CE-loss gap,
own-objective loss ratios, and HEP alpha gains so the next PC objective variant
can be chosen before another promotion-style gate.

The first PC objective variant adds a small supervised CE anchor to the PC
logit-MSE loss while preserving the objective-gate setting that disables the
support-stress preset and keeps the promoted temporal clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_pc_anchor_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_pc_anchor_temporal_clipped_objective_gate \
  --out results/comparisons/validation_pc_anchor_temporal_clipped_objective_gate/artifact_check_local.json
```

The current local evidence passes artifacts and invariants. The anchored PC
variant closes nearly all of the unanchored PC supervised-CE HEP loss gap, but
supervised CE remains slightly better under the same validation gate.

The matching Colab validation uses the same three configs and writes the
Colab-prefixed artifact tree:

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

After matching local and Colab anchored-PC artifacts exist, write the
command-driven anchored-PC objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report anchored-pc-residual-objective-decision \
  --out results/reports/anchored_pc_residual_objective_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current local and Colab artifacts, anchored PC closes most of the unanchored PC
supervised-CE HEP loss gap but still does not beat supervised CE, so the report
stops PC residual-objective validation under the current gate.

The first non-PC residual objective variant adds a small confidence penalty to
supervised CE while preserving the objective-gate setting that disables the
support-stress preset and keeps the promoted temporal clipped HEP path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_confidence_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_confidence_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/validation_confidence_penalty_temporal_clipped_objective_gate/artifact_check.json
```

The current local evidence passes artifacts and invariants. The confidence
penalty improves its own configured objective by a similar ratio to supervised
CE, but supervised CE still has the lower best temporal-clipped HEP CE loss in
this first validation comparison.

The matching Colab validation uses the same objective-gate pair and writes the
Colab-prefixed artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_confidence_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and Colab confidence-penalty artifacts exist, write the
command-driven confidence-penalty objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report confidence-penalty-residual-objective-decision \
  --out results/reports/confidence_penalty_residual_objective_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current local and Colab artifacts, confidence penalty improves its own final
residual training loss but is still worse than supervised CE on best
temporal-clipped HEP supervised loss, so the report stops confidence-penalty
objective validation under the current gate.

The next non-PC residual objective variant adds a small target-logit margin
penalty to supervised CE while preserving the same objective-discriminative
temporal clipped HEP gate:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_margin_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_margin_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/validation_margin_penalty_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and Colab margin-penalty artifacts exist, write the
command-driven margin-penalty objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report margin-penalty-residual-objective-decision \
  --out results/reports/margin_penalty_residual_objective_decision
```

The label-smoothing objective-gate validation adds the next non-PC residual
objective variant to the same command-driven Colab validation path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_label_smoothing_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and Colab label-smoothing artifacts exist, write the
command-driven label-smoothing objective decision report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report label-smoothing-residual-objective-decision \
  --out results/reports/label_smoothing_residual_objective_decision
```

The focal objective-gate validation adds the next non-PC residual objective
variant to the same objective-discriminative temporal clipped HEP gate:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_focal_temporal_clipped_objective_gate \
  --out results/comparisons/validation_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_validation_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_focal_temporal_clipped_objective_gate/artifact_check.json
```

After matching local and Colab focal artifacts exist, write the command-driven
focal objective decision report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report focal-residual-objective-decision \
  --out results/reports/focal_residual_objective_decision
```

The broader focal objective-gate check moves outside the current char
validation setting while preserving the objective-discriminative temporal
clipped HEP path. It uses sequence length `96`, hidden dimension `64`, `16`
residual columns, `4` HEP settling steps, `30` training steps, and disables the
support-stress preset so CE/HEP behavior is measured on learned residual values:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/extended_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/extended_focal_temporal_clipped_objective_gate \
  --out results/comparisons/extended_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_extended_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_extended_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_extended_focal_temporal_clipped_objective_gate/artifact_check.json
```

The next focal objective-gate scale check uses the larger char-level setting
from the temporal-clipped promotion gate while preserving the
objective-discriminative path. It uses sequence length `128`, hidden dimension
`96`, `24` residual columns, `4` HEP settling steps, `50` training steps, and
disables the support-stress preset so CE/HEP behavior is measured on learned
residual values:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/larger_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation should use the same pair under the Colab-prefixed
artifact tree after the config additions are available in GitHub:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_larger_focal_temporal_clipped_objective_gate/artifact_check.json
```

The next tokenized focal objective-gate scale check mirrors the non-char
tokenized temporal promotion-gate setting while preserving the
objective-discriminative path. It uses deterministic word-token IDs with
sequence length `64`, hidden dimension `96`, `24` residual columns, `4` HEP
settling steps, `50` training steps, and disables the support-stress preset:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/token_larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/token_larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/token_larger_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate/artifact_check.json
```

The next xlarge char-level focal objective-gate check increases the char
setting to sequence length `160`, hidden dimension `128`, `32` residual
columns, `4` HEP settling steps, and `60` training steps while keeping the
objective-discriminative path and disabling the support-stress preset:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/char_xlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/char_xlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/char_xlarge_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/artifact_check.json
```

The next xxlarge char-level focal objective-gate check increases the char
setting to sequence length `192`, hidden dimension `160`, `40` residual
columns, `4` HEP settling steps, and `70` training steps while keeping the
objective-discriminative path and disabling the support-stress preset:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate/artifact_check_local.json
```

The matching Colab validation uses the same pair under the Colab-prefixed
artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate/artifact_check.json
```

After the validation, extended, larger, tokenized larger, xlarge, and xxlarge
local/Colab focal objective artifacts exist, define the next focal promotion or
stop gate without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report focal-residual-objective-promotion-gate \
  --out results/reports/focal_residual_objective_promotion_gate
```

The report writes `decision_report.json` and `decision_report.md`. It consumes
the focal residual-objective decision report and records the next required
evidence before any default residual-objective change: a seed-2 repeat of the
xxlarge char focal-vs-supervised objective gate and a seed-2 repeat of the
tokenized larger focal-vs-supervised objective gate, each with local and Colab
artifact-backed checks. This report defines the gate only; it does not promote
focal CE by itself.

The seed-2 xxlarge char repeat uses the same objective-discriminative temporal
clipped path and scale as the seed-1 xxlarge gate, with only the seed and
experiment IDs changed:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2/artifact_check_local.json
```

The matching seed-2 Colab validation uses the same pair under the
Colab-prefixed artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/colab_char_xxlarge_focal_temporal_clipped_objective_gate_seed2/artifact_check.json
```

The seed-2 tokenized larger repeat mirrors the seed-1 tokenized objective gate
with deterministic word-token IDs and the same scale:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/token_larger_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/token_larger_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/token_larger_focal_temporal_clipped_objective_gate_seed2/artifact_check_local.json
```

The matching seed-2 Colab validation uses the same pair under the
Colab-prefixed artifact tree:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml \
  --config configs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2 \
  --out results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2/artifact_check.json
```

After the seed-2 xxlarge-char and tokenized larger local/Colab repeat artifacts
exist, decide whether the focal promotion/stop gate is satisfied without
rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report focal-residual-objective-promotion-gate-satisfaction \
  --out results/reports/focal_residual_objective_promotion_gate_satisfaction
```

The report writes `decision_report.json` and `decision_report.md`. It promotes
focal CE only if all four repeat comparisons pass artifacts/invariants and
focal CE beats supervised CE on best temporal-clipped supervised CE HEP loss in
each comparison; otherwise it stops focal validation under the current gate.

The earlier label-free temporal validation uses the same command-driven
harness and writes the artifact tree under the Colab-prefixed validation path:

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

After a completed Colab run and local artifact extraction, inspect the
extracted artifact tree and write the validation temporal decision report
without rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_validation_colab_decision
```

A further opt-in temporal clipped HEP support-stress check moves outside the
current char validation setting by using sequence length `96`, `16` residual
columns, `4` HEP settling steps, and `30` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_support_stress_clipped.yaml \
  --config configs/char_extended_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_extended_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_extended_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/extended_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/extended_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/extended_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/extended_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/extended_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_extended_local_decision
```

This keeps temporal consistency as an opt-in candidate while checking whether
the label-free signal survives a longer-context support-stress probe before any
default-promotion decision.

After the seed-smoke, validation, and extended local/Colab temporal decision
reports exist, write the command-driven cross-scale aggregate report without
rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep-cross-scale-aggregate \
  --out results/reports/temporal_clipped_hep_cross_scale_aggregate
```

The cross-scale aggregate fails closed if any expected decision report is
missing, fails, does not select temporal consistency, lacks an accepted nonzero
temporal alpha inside the stability budgets, or omits a seed-smoke, validation,
or extended local/Colab evidence pair. It can select temporal consistency
across the current char support-stress evidence, but still blocks default
promotion until a broader promotion gate is defined and run.

After the cross-scale aggregate exists, define the next temporal clipped HEP
promotion gate without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep-promotion-gate \
  --out results/reports/temporal_clipped_hep_promotion_gate
```

The promotion-gate report consumes the cross-scale aggregate and records the
next required evidence before any default support-stress mitigation change: one
larger char-level temporal-vs-entropy-vs-guided clipped support-stress
comparison and one non-char tokenized comparison, each with local and Colab
artifact-backed decisions, passing artifact checks, nonzero support-repick
evidence, and an accepted nonzero temporal alpha inside the ordinary-logit and
pinned-vs-repicked budgets. This report defines the gate only; it does not
promote temporal clipped HEP by itself.

The larger char-level promotion-gate comparison uses sequence length `128`,
hidden dimension `96`, `24` residual columns, `4` HEP settling steps, and `50`
training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_support_stress_clipped.yaml \
  --config configs/char_larger_hep_support_stress_entropy_clipped.yaml \
  --config configs/char_larger_hep_support_stress_temporal_clipped.yaml \
  --config configs/char_larger_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/larger_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
```

The non-char tokenized promotion-gate comparison uses deterministic word-token
IDs rather than character IDs while keeping the same command-driven HEP
candidate structure. It uses sequence length `64`, hidden dimension `96`, `24`
residual columns, `4` HEP settling steps, and `50` training steps:

```bash
python -m relaleap.experiments.compare \
  --config configs/token_larger_hep_support_stress_clipped.yaml \
  --config configs/token_larger_hep_support_stress_entropy_clipped.yaml \
  --config configs/token_larger_hep_support_stress_temporal_clipped.yaml \
  --config configs/token_larger_hep_support_stress_guided_clipped.yaml \
  --out results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_token_larger_local_decision
```

The matching GitHub-backed Colab larger-char promotion-gate check writes the
same artifact-backed comparison under the Colab-prefixed path:

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

After extracting the Colab artifact bundle locally, inspect the tree and write
the larger-char temporal decision report without rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_larger_colab_decision
```

The matching GitHub-backed Colab non-char tokenized promotion-gate check writes
the same artifact-backed comparison under the Colab-prefixed path:

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

After extracting the Colab artifact bundle locally, inspect the tree and write
the tokenized temporal decision report without rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --out results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep \
  --comparison-dir results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep \
  --artifact-check results/comparisons/colab_token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/artifact_check_local.json \
  --out results/reports/temporal_clipped_hep_token_larger_colab_decision
```

After the larger-char and tokenized local/Colab decision reports exist, write
the command-driven promotion-gate satisfaction report without rerunning
experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-clipped-hep-promotion-gate-satisfaction \
  --out results/reports/temporal_clipped_hep_promotion_gate_satisfaction
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the promotion-gate definition report passes and the larger-char
and tokenized local/Colab reports all pass, select temporal consistency, show
nonzero support repicking, and include an accepted nonzero temporal alpha inside
the ordinary-logit and pinned-vs-repicked budgets. A passing report satisfied
the defined promotion gate and led to the explicit default support-stress
mitigation change to temporal clipped HEP.

After the default support-stress mitigation has been promoted, define the next
residual-layer learning gate without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report post-promotion-residual-learning-gate \
  --config-path configs/char_smoke_hep_support_stress.yaml \
  --out results/reports/post_promotion_residual_learning_gate
```

The report fails closed unless the temporal clipped HEP promotion-gate
satisfaction report passes and `configs/char_smoke_hep_support_stress.yaml`
contains the promoted temporal-consistency settling objective with the `0.01`
clip. A passing report defines, but does not run or promote, the next gate:
compare supervised residual updates against the existing PC-style residual
objective under the promoted temporal clipped support-stress default, with
local and Colab artifact-backed decisions before any residual-objective change.

The first post-focal residual-learning direction is an opt-in supervised CE
objective with a small train-time temporal-consistency regularizer. It keeps
the objective-gate setting that disables the support-stress preset and uses the
promoted temporal clipped HEP inference path:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_temporal_consistency_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_temporal_consistency_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_temporal_consistency_temporal_clipped_objective_gate \
  --out results/comparisons/validation_temporal_consistency_temporal_clipped_objective_gate/artifact_check_local.json
```

The first local validation artifact passes invariants and artifact checks, but
the temporal-consistency training regularizer is not discriminative at this
scale: its best temporal-clipped HEP supervised CE loss ties the supervised CE
baseline within the current deterministic precision.

A stronger local-only train-time temporal-consistency sweep keeps the same
validation objective-gate setting and compares the original `0.01` weight with
`0.05`, `0.1`, and `0.2` before spending Colab time:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_temporal_consistency_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_temporal_consistency_w005_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_temporal_consistency_w010_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate \
  --out results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/artifact_check_local.json
```

The next bounded local-only check repeats the same temporal-consistency weight
sweep at the extended char objective-gate scale: sequence length `96`, hidden
dimension `64`, `16` residual columns, `4` HEP settling steps, and `30`
training steps.

```bash
python -m relaleap.experiments.compare \
  --config configs/char_extended_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_temporal_consistency_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_temporal_consistency_w005_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_temporal_consistency_w010_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_extended_temporal_consistency_w020_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/extended_temporal_consistency_weight_sweep_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/extended_temporal_consistency_weight_sweep_temporal_clipped_objective_gate \
  --out results/comparisons/extended_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/artifact_check_local.json
```

After the validation and extended local sweeps complete, write the
command-driven train-time temporal-consistency residual-objective decision
report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report temporal-consistency-residual-objective-decision \
  --out results/reports/temporal_consistency_residual_objective_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current local-only policy, temporal-consistency regularizer validation only
continues if every regularized sweep run beats the supervised CE baseline by a
material best temporal-clipped HEP supervised CE-loss margin. Tiny deterministic
margins stop the variant before spending Colab time.

After the residual-objective gate, anchored-PC, confidence-penalty,
margin-penalty, label-smoothing, focal promotion/stop gate, and train-time
temporal-consistency reports have all stopped under their current policies,
select the next residual-learning direction without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-learning-next-direction \
  --out results/reports/residual_learning_next_direction
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the expected completed decision reports are present and passing.
When it passes, it keeps supervised CE as the default residual objective and
selects a residual capacity/support diagnostic as the next bounded research
direction rather than another CE-adjacent objective variant.

Define that diagnostic as a local-only gate before spending Colab time:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-capacity-support-diagnostic-gate \
  --out results/reports/residual_capacity_support_diagnostic_gate
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the residual-learning next-direction report passes and the config
matrix keeps the promoted temporal-clipped supervised CE validation harness
fixed while varying only residual column capacity and top-k support width.
When the gate passes, run the local diagnostic comparison:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate \
  --out results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate/artifact_check_local.json
```

After the local comparison and artifact check complete, decide whether the
widened-support result merits matching Colab validation:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-capacity-support-diagnostic-decision \
  --comparison-dir results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate \
  --artifact-check results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate/artifact_check_local.json \
  --out results/reports/residual_capacity_support_diagnostic_decision
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the local comparison and artifact contract pass, all four
capacity/support variants are present, and widened support is the best local
variant with an accepted nonzero temporal-clipped HEP alpha inside the ordinary
logit-delta budget. When it passes, the next step is the matching real-Chrome
Colab bridge run that emits
`results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate`.
After extracting that Colab bundle locally, inspect it and write the paired
local/Colab confirmation report without rerunning experiments:

```bash
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate \
  --out results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report residual-capacity-support-diagnostic-colab-decision \
  --out results/reports/residual_capacity_support_diagnostic_colab_decision
```

The paired report fails closed unless both local and Colab artifact-backed
comparisons pass and both select widened support as the best variant. A failing
paired report is treated as a divergence diagnosis input, not a promotion gate.
When it passes, define the broader support-width validation gate without
rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-support-width-validation-gate \
  --out results/reports/residual_support_width_validation_gate
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the paired local/Colab capacity-support diagnostic report passes
and the larger char/tokenized config matrix keeps supervised CE, promoted
temporal-clipped HEP, column capacity, and support-stress settings fixed while
only widening top-k support from `1` to `2` within each scale. When the gate
passes, run its recorded local comparison and artifact check:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate \
  --out results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate/artifact_check_local.json
```

After seed-2 repeat evidence and seed-3 local/Colab promotion-gate artifacts
exist, write the command-driven support-width promotion-gate satisfaction report
without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --report residual-support-width-promotion-gate-satisfaction \
  --out results/reports/residual_support_width_promotion_gate_satisfaction
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the promotion-gate definition passes, both seed-3 local and Colab
artifact trees pass, the run identities are seed-3, and top-k `2` improves
ordinary alpha-0 and final supervised CE loss over top-k `1` at both
larger-char and tokenized scales.

After the support-width promotion gate is satisfied, the default larger-char
and tokenized objective-gate configs use residual-column support top-k `2`.
Verify the promoted default locally with a fresh command-driven comparison and
artifact check:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/post_promotion_support_width_default_temporal_clipped_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/post_promotion_support_width_default_temporal_clipped_objective_gate \
  --out results/comparisons/post_promotion_support_width_default_temporal_clipped_objective_gate/artifact_check_local.json
```

With top-k `2` promoted, the bounded train-time temporal-consistency
regularizer retest compares the previous strongest local weight, `0.2`, against
the current supervised CE defaults at the larger char and tokenized scales:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_temporal_consistency_w020_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_temporal_consistency_w020_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/post_support_width_temporal_consistency_w020_larger_token_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/post_support_width_temporal_consistency_w020_larger_token_objective_gate \
  --out results/comparisons/post_support_width_temporal_consistency_w020_larger_token_objective_gate/artifact_check_local.json
```

The local evidence passes artifacts and invariants but keeps supervised CE as
the better residual objective at both promoted scales: char supervised best HEP
CE `3.14853334` beats temporal-consistency `3.15985703`, and tokenized
supervised alpha-0 CE `3.48037004` beats temporal-consistency `3.59536052`.

After the post-support-width temporal-consistency retest stops, define the next
non-CE-adjacent residual-learning gate under the promoted supervised CE,
temporal-clipped HEP, and top-k `2` support defaults:

```bash
python -m relaleap.experiments.decision_report \
  --report post-support-width-residual-learning-gate \
  --out results/reports/post_support_width_residual_learning_gate
```

The report writes `decision_report.json` and `decision_report.md`. It fails
closed unless the support-width promotion report passes with selected top-k `2`
and the focused larger-char/tokenized config matrix keeps supervised CE,
temporal-clipped HEP, `support_stress_preset: false`, and `top_k: 2` fixed.
When it passes, run its recorded local residual-capacity comparison:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_larger_capacity_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_hep_temporal_clipped_objective_gate.yaml \
  --config configs/token_larger_capacity_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/post_support_width_capacity_larger_token_objective_gate
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/post_support_width_capacity_larger_token_objective_gate \
  --out results/comparisons/post_support_width_capacity_larger_token_objective_gate/artifact_check_local.json
```

The post-support-width residual-capacity decision stopped capacity validation
under mixed local/Colab evidence and selected support-width deconfounding plus
exhaustive support audit as the next non-CE-adjacent direction. The first local
support-width deconfounding matrix reuses the validation-scale 2x2 controls:
12 versus 24 columns crossed with top-k `1` versus top-k `2`, while holding
supervised CE, temporal-clipped HEP, and `support_stress_preset: false` fixed:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_validation_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_capacity_hep_temporal_clipped_objective_gate.yaml \
  --config configs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/comparisons/support_width_deconfounding_validation_audit
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_width_deconfounding_validation_audit \
  --out results/comparisons/support_width_deconfounding_validation_audit/artifact_check_local.json
python -m relaleap.experiments.decision_report \
  --report support-width-deconfounding-audit \
  --comparison-dir results/comparisons/support_width_deconfounding_validation_audit \
  --artifact-check results/comparisons/support_width_deconfounding_validation_audit/artifact_check_local.json \
  --out results/reports/support_width_deconfounding_validation_audit
```

The local audit report passes and selects the matching real-Chrome Colab repeat
as the next step. At this scale, top-k `2` improves best temporal-clipped HEP
CE and expands support utilization from one used column to 11 used columns,
while doubling columns at top-k `1` leaves the audit collapsed onto one column.

After matching local and Colab support-width deconfounding matrices exist, run
the first bounded exhaustive support audit for the validation-scale 12-column
top-k `2` support-width config:

```bash
python -m relaleap.experiments.support_audit \
  --config configs/char_validation_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/validation_support_wide_exhaustive_support
python -m relaleap.experiments.decision_report \
  --report exhaustive-support-audit \
  --comparison-dir results/audits/validation_support_wide_exhaustive_support \
  --out results/reports/validation_support_wide_exhaustive_support_audit
```

The audit retrains the configured residual adapter, evaluates all 12 singleton
supports and all 66 two-column supports on the fixed in-repo validation batch,
and writes `summary.json`, `support_losses.csv`, `pairwise_synergy.csv`,
`router_target_diagnostic.csv`, `router_target_nonlinear_diagnostic.csv`,
`router_target_contextual_diagnostic.csv`, and `notes.md`. The summary reports
per-token oracle-support regret, best global fixed support, dominant router
support, one-swap recovery, support-loss
distribution, pairwise synergy leaders, the existing router support audit, and
linear, small-MLP, and contextual small-MLP router-target probes that train
against the per-token oracle support pair on even flattened token positions and
report holdout odd-position recovery of the oracle gap. The contextual probe
adds normalized token position plus immediate previous/next hidden-neighborhood
features. The decision report fails closed on missing audit artifacts and
selects whether the next branch should target router support selection, column
redundancy, or pairwise composition.

A paired pinned-support stress config uses the same support-stress preset while
pinning settling updates to the ordinary-pass support:

```bash
python -m relaleap.experiments.compare \
  --config configs/char_smoke_hep_support_stress.yaml \
  --config configs/char_smoke_pinned_hep_support_stress.yaml \
  --out results/comparisons/support_stress_pinned_vs_repicked
python -m relaleap.experiments.check_artifacts \
  --comparison-dir results/comparisons/support_stress_pinned_vs_repicked
```

The comparison summary and notes include per-run `pinned_support`,
`support_stress`, `support_instability`, and HEP sweep support diagnostics.

After a completed local or Colab support-stress comparison, write the
command-driven pinned-support decision report without rerunning experiments:

```bash
python -m relaleap.experiments.decision_report \
  --comparison-dir results/comparisons/colab_support_stress_pinned_vs_repicked \
  --artifact-check results/comparisons/colab_support_stress_pinned_vs_repicked/artifact_check_local.json \
  --out results/reports/pinned_support_decision
```

The report writes `decision_report.json` and `decision_report.md`. Under the
current default policy, pinned support should remain opt-in unless the evidence
artifacts pass and a pinned nonzero HEP alpha improves loss over alpha 0 while
staying within the ordinary-logit delta budget of `0.1`.

Colab should be treated as a temporary GPU runner, not the source of truth. The GitHub repo, config files, and run artifacts are the source of truth.

## Temporary Colab Bridge

The first GitHub-backed Colab notebook is:

```text
https://colab.research.google.com/github/bgoertzel-sing/relaleap/blob/main/notebooks/relaleap_colab_smoke.ipynb
```

For a brittle command-line/browser automation bridge, see:

```text
docs/colab_bridge.md
tools/colab_playwright_runner.py
```

Use a modern Python environment for that bridge. The old conda `base` Python
3.7 environment cannot install current Playwright wheels.

One-command setup:

```bash
bash tools/setup_colab_bridge.sh
```
