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

After a promoted support-wide contextual-router run, inspect whether remaining
dead or underloaded columns look like router-load skew or learned-value
redundancy without rerunning the exhaustive oracle-support matrix:

```bash
python -m relaleap.experiments.column_redundancy \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_support_wide_promoted_default_column_redundancy
```

This writes `summary.json`, `column_loads.csv`,
`column_pair_similarity.csv`, and `notes.md` with column support counts, load
entropy, effective-column count, column-value norms, and pairwise column-value
similarities.

To probe whether dead columns in the promoted tokenized support-wide setting can
be recruited by a small differentiable router load-balancing intervention
without hurting alpha-0 CE, run:

```bash
python -m relaleap.experiments.dead_column_probe \
  --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_support_wide_promoted_default_dead_column_probe
```

This opt-in diagnostic writes `summary.json`, `variant_metrics.csv`, and
`notes.md` for the baseline and several load-balance weights. It does not change
the promoted default router policy or any checked Phase 0 baseline.

After a causal-column fingerprint audit that includes the rank-matched
contextual top-k-1 bracket, compare promoted top-k-2 and rank-matched top-k-1
on the existing token/position-stratified intervention rows without retraining:

```bash
python -m relaleap.experiments.matched_strata_intervention_audit \
  --audit-dir results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1 \
  --out results/audits/token_larger_rank_matched_topk1_vs_topk2_matched_strata_intervention
```

This post-hoc audit writes `summary.json`, `matched_strata.csv`, and `notes.md`.
It fails closed when the top-k-2 fixed-pair rows or rank-matched top-k-1
fixed-singleton rows are absent, and keeps the causal-cooperation claim
conservative unless top-k-2 wins on matched-strata synergy, fixed-support
cleanliness, functional churn, and router CE.

To decide whether the existing causal-audit artifacts are already sufficient for
the next no-training top-k-2 versus rank-matched top-k-1 deconfounding audit, run:

```bash
python -m relaleap.experiments.causal_audit_coverage_report
```

This source-artifact coverage report writes
`results/reports/token_larger_causal_audit_coverage/decision_report.json` and
`.md`. It lists the variants, intervention rows, strata, row granularity,
residual/gain fields, support/functional churn fields, and random/dense/rank
controls available across the existing source artifacts, then emits one of
`existing_artifacts_sufficient_for_next_no_training_audit`,
`specific_missing_fields_require_artifact_extension`, or
`new_training_required_for_deconfounded_causal_matrix`. When the post-stop
causal-bracket decision report exists, the coverage report consumes it as the
later source of truth and emits `rank_matched_topk1_active_post_stop_bracket`
instead of sending the loop back to a top-k-2 causal-cooperation audit. In that
state, rank-matched contextual top-k-1 is the active local causal bracket, while
top-k-2 causal-cooperation and support-frequency candidate-percentile claims
remain blocked unless no-fallback support-frequency controls become identified
and supportive.
The causal-column fingerprint audit now also writes
`per_token_pair_interventions.csv` with batch/position/token indices,
residual-norm and residual-gain bins, support-frequency fields, and an
active-rank proxy. For top-k-2 fixed-pair interventions it also logs
singleton-left/singleton-right per-token losses and gains, pair gain, and
`pair_synergy = pair_gain - singleton_left_gain - singleton_right_gain`, so the
next no-training residual-norm/active-rank deconfounding audit can test direct
per-token synergy without new training. When run with
`--include-rank-matched-topk1`, the active top-k-1 variant also emits
deterministic random singleton controls and exhaustive singleton rows for the
same context keys.

After the coverage report emits
`existing_artifacts_sufficient_for_next_no_training_audit`, run the no-training
deconfounded intervention audit:

```bash
python -m relaleap.experiments.deconfounded_intervention_audit
```

This writes
`results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention/summary.json`,
`matched_deconfounded_strata.csv`, `paired_exact_context_deltas.csv`, and
`notes.md`. It compares promoted contextual top-k-2 with rank-matched
contextual top-k-1 by first pairing exact
`batch_index`/`position_index`/`token_index`/`target_token` contexts, then
summarizing those paired rows inside position-bin, token-class,
residual-norm-bin, residual-gain-bin, and router-support-count-bin strata. The
active-rank proxy is reported as a bracket dimension rather than exact-matched
because top-k-2 and top-k-1 have structurally different active-rank proxies in
the current artifact. CE is treated as a guardrail with a default tolerance of
`0.05`, and the audit reports direct deconfounded per-token pair synergy
separately from the stricter comparative causal-cooperation claim. Top-k-2
internal pair synergy is not sufficient unless the incremental top-k-2 pair gain
also beats the matched rank-matched top-k-1 singleton bracket across at least
`80%` of matched strata and the fixed-support/functional-churn cleanliness gates
also pass.

After a causal contextual-router support audit blocks promotion because the
causal router wins CE but fails oracle-regret or functional-churn gates, run the
bounded train-time support-stability regularization probe without changing
defaults:

```bash
python -m relaleap.experiments.causal_contextual_router_regularization_probe \
  --config configs/token_larger_support_wide_causal_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_causal_contextual_router_regularization_probe
```

This writes `summary.json`, `fold_metrics.csv`, `aggregate_metrics.csv`,
`variant_gate.csv`, `control_metrics.csv`, `support_counts.csv`, and `notes.md`.
It compares the unregularized causal contextual top-k-2 router, linear top-k-2,
adjacent router-score smoothness variants, soft support-load balance variants,
and train-fold oracle-target support variants under the same sequence-heldout
oracle-regret and functional-churn audit metrics. The oracle-target penalty is
a train-time diagnostic regularizer computed only on the training fold. A
passing variant is only a candidate for backend validation; it does not promote
a router default.

After the deconfounded audit closes comparative top-k-2 causal cooperation,
confirm the active local causal bracket without rerunning training:

```bash
python -m relaleap.experiments.decision_report \
  --report active-rank-matched-topk1-causal-bracket-audit
```

This writes
`results/reports/token_larger_active_rank_matched_topk1_causal_bracket_audit/decision_report.json`
and `.md`. It consumes the exact-context deconfounded audit, records
rank-matched contextual top-k-1 as the primary local causal bracket, keeps
promoted top-k-2 as a reference condition only, and leaves both top-k-2
causal-cooperation and support-frequency percentile claims closed.

After the active rank-matched contextual top-k-1 bracket is confirmed, build the
local top-k-1 separability packet without retraining:

```bash
python -m relaleap.experiments.active_topk1_causal_separability_audit
```

This writes
`results/audits/token_larger_active_rank_matched_topk1_causal_separability/summary.json`,
`topk1_separability_by_stratum.csv`, `topk1_separability_by_context.csv`, and
`notes.md`. It consumes the exact-context deconfounded audit and records
rank-matched top-k-1 singleton gain, fixed-support loss delta, fixed-support
logit churn, and residual-stream churn while retaining top-k-2 only as the
closed reference condition. A passing packet establishes measurement coverage,
not a separability claim by itself.

After the active top-k-1 separability packet exists, run the bounded local
retention/churn probe for the same active bracket:

```bash
python -m relaleap.experiments.active_topk1_retention_churn_probe
```

This writes
`results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe/summary.json`,
`variant_metrics.csv`, `phase_metrics.csv`, and `notes.md`. It reruns the
local anchor/transfer retention microtest, requires the active top-k-1
separability packet as source evidence, and compares rank-matched contextual
top-k-1 against the promoted contextual top-k-2 reference and a norm-matched
dense active-rank control. A passing probe supports top-k-1 as the local
retention/churn bracket, but does not establish singleton causal separability.
The probe also records finite-update order-sensitivity evidence by comparing
the final functions from A-to-B and B-to-A training orders on both anchor and
transfer slices.

To check whether that low-churn signal is seed-stable before treating it as a
settled bracket result, run the seed-2 repeat:

```bash
python -m relaleap.experiments.active_topk1_retention_churn_probe \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2.yaml \
  --out results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2
```

This uses the same active top-k-1 separability packet as source evidence and
only changes the local retention/churn microtest seed.

After seed-1 and seed-2 active top-k-1 retention/churn probe packets exist,
summarize their local stability without rerunning training:

```bash
python -m relaleap.experiments.active_topk1_retention_churn_summary
```

This writes
`results/reports/token_larger_active_rank_matched_topk1_retention_churn_stability/summary.json`,
`probe_metrics.csv`, and `notes.md`. It requires both probe packets to pass
their active-bracket gates and records rank-matched contextual top-k-1 as a
local retention/churn stability bracket only; it keeps the negative singleton
gain caveat from the source separability packet and does not reopen the closed
top-k-2 causal-cooperation claim.

To turn the completed seed-1/seed-2 probe packets into a fail-closed
functional-retention decision packet without rerunning training:

```bash
python -m relaleap.experiments.active_topk1_functional_retention_audit
```

This writes
`results/reports/token_larger_active_topk1_functional_retention_audit/summary.json`,
`packet_metrics.csv`, and `notes.md`. It separates support identity churn,
functional/logit churn, causal singleton-gain status, and CE guardrails, and
uses explicit decision labels such as `functional_retention_bracket_only`,
`functional_retention_claim_supported`, `blocked_by_negative_singleton_gain`,
`context_gated_singleton_efficacy_with_offcontext_interference`, and
`blocked_by_control_match_failure`. When the selected/oracle/off-context
singleton reconciliation audit is present, the functional-retention report uses
that newer source instead of the stale global negative-singleton blocker. The
current packets are one-order transfer probes plus A-to-B versus B-to-A
finite-update commutator evidence; the report can treat a favorable commutator
as bracket evidence, but still blocks a broad reusable singleton
causal-retention claim when the reconciliation shows off-context singleton
interference.

After the fresh RunPod task-free retention microtest packets exist for seed 1
and seed 2, synthesize the retention, finite-update, support-selection,
functional-churn, and causal/deconfounded source artifacts without rerunning
training:

```bash
python -m relaleap.experiments.promoted_topk2_retention_synthesis_gate
```

This writes
`results/reports/token_larger_promoted_topk2_retention_synthesis_gate/summary.json`,
`source_rows.csv`, `retention_seed_metrics.csv`, and `notes.md`. It decides
whether the next evidence step should be another fresh retention seed or a
bounded support-stability/finite-update mitigation probe while retaining the
promoted top-k-2, rank-matched top-k-1, random fixed top-k-2, and dense
active-rank controls.

To materialize the promoted contextual top-k-2 causal-adequacy matrix as a
single fail-closed packet before choosing any default-router interpretation:

```bash
python -m relaleap.experiments.promoted_topk2_causal_adequacy_matrix
```

This writes
`results/reports/token_larger_promoted_topk2_causal_adequacy_matrix/summary.json`,
`causal_adequacy_matrix.csv`, `source_rows.csv`, and `notes.md`. It is a
no-training synthesis over command-generated artifacts and compares promoted
contextual top-k-2 with rank-matched contextual top-k-1, random fixed top-k-2,
and dense active-rank controls. The gate treats CE as a guardrail and requires
retention/churn, finite-update commutator, deconfounded intervention
cleanliness, and oracle-support-regret evidence before allowing any
causal-adequacy claim; otherwise top-k-2 remains only the predictive
support-routing default and top-k-1 remains the retention/churn control.

To run that bounded mitigation probe through the command-driven microtest
harness, including router-freeze and update-clipped top-k-2 variants against
the same controls:

```bash
python -m relaleap.experiments.promoted_topk2_retention_mitigation_probe
```

This writes
`results/audits/token_larger_promoted_topk2_retention_mitigation_probe/summary.json`,
`variant_metrics.csv`, `phase_metrics.csv`, `mitigation_rows.csv`, and
`mitigation_notes.md`. The gate requires a mitigation to materially reduce
absolute anchor commutator logit MSE while retaining transfer improvement and
support usage, so a support-collapse or transfer-destroying variant is rejected
even if its support churn falls.

After a negative mitigation gate, decompose the promoted contextual top-k-2
finite-update packet into full, router-only, and value-only transfer updates
under the same A/B versus B/A microtest protocol:

```bash
python -m relaleap.experiments.promoted_topk2_update_decomposition_audit
```

This writes
`results/audits/token_larger_promoted_topk2_update_decomposition_audit/summary.json`,
`variant_metrics.csv`, `phase_metrics.csv`, `decomposition_rows.csv`, and
`decomposition_notes.md`. The audit keeps full anchor-slice training fixed,
then restricts the second-slice update group to router-only or value-only to
identify whether order sensitivity is mainly router-update, value-update, or
mixed before selecting a mitigation family.

After the decomposition audit identifies value-update dominated order
sensitivity, run the bounded value-update mitigation gate before returning to
router-policy changes:

```bash
python -m relaleap.experiments.promoted_topk2_value_mitigation_gate
```

This writes
`results/audits/token_larger_promoted_topk2_value_mitigation_gate/summary.json`,
`variant_metrics.csv`, `phase_metrics.csv`, `value_mitigation_rows.csv`, and
`value_mitigation_notes.md`. The gate compares promoted contextual top-k-2
against value-gradient-clipped and value-update-scaled top-k-2 variants while
preserving the rank-matched top-k-1, random fixed top-k-2, and dense active-rank
controls. A candidate must materially reduce absolute anchor commutator logit
MSE while retaining transfer improvement and support usage.

After the shortcut-decision selector chooses the commutator-aware value-penalty
branch, run the bounded local probe:

```bash
python -m relaleap.experiments.promoted_topk2_commutator_value_penalty_probe \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe
```

This writes
`results/audits/token_larger_promoted_topk2_commutator_value_penalty_probe/summary.json`,
`variant_metrics.csv`, `phase_metrics.csv`, `per_token_commutator.csv`,
`commutator_value_penalty_rows.csv`, and
`commutator_value_penalty_notes.md`. It compares promoted contextual top-k `2`
against two residual-change penalty weights while preserving the rank-matched
top-k `1`, random fixed top-k `2`, and dense active-rank controls. A candidate
must materially reduce absolute anchor commutator logit MSE while retaining
transfer improvement, support usage, and anchor CE drift.

After simple value scaling/clipping, low-rank value updates, and the
commutator-aware value penalty all fail their promoted top-k-2 commutator
mitigation gates, select exactly one next mitigation branch without rerunning
training:

```bash
python -m relaleap.experiments.promoted_topk2_mitigation_branch_selector
```

This writes
`results/reports/token_larger_promoted_topk2_mitigation_branch_selector/summary.json`,
`source_rows.csv`, `candidate_actions.csv`, and `notes.md`. It consumes the
shortcut decision, finite-update order-control report, simple value mitigation
gate, low-rank value gate, and commutator-value-penalty probe. The selector
fails closed on missing or inconsistent packets and chooses exactly one of
`router_policy_mitigation_probe` or `explicit_order_averaging_mitigation_probe`;
order averaging remains a mitigation candidate only, not a causal-cooperation
claim.

After the selector chooses explicit order averaging, record it as a fail-closed
diagnostic and choose the next branch without rerunning training:

```bash
python -m relaleap.experiments.promoted_topk2_explicit_order_averaging_mitigation_probe \
  --out results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe
```

This writes
`results/audits/token_larger_promoted_topk2_explicit_order_averaging_mitigation_probe/summary.json`,
`source_rows.csv`, `order_averaging_rows.csv`, and `notes.md`. The report
consumes the mitigation-branch selector plus finite-update order-control
packet, treats order averaging as diagnostic-only and not a promoted
architecture path, and emits the exact next command for the router-policy
mitigation branch when the diagnostic source evidence is present.

After order averaging points back to router-policy mitigation, run the
fail-closed paired intervention/decomposition gate before spending GPU time:

```bash
python -m relaleap.experiments.promoted_topk2_router_policy_mitigation_probe \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe
```

This writes
`results/audits/token_larger_promoted_topk2_router_policy_mitigation_probe/summary.json`,
`source_rows.csv`, `router_policy_rows.csv`, `interpretation_rows.csv`,
`router_policy_interventions.csv`, and `notes.md`. By default the command runs
a bounded local retention/churn microtest over the promoted contextual top-k
`2` packet and compares dynamic routing with pinned pre-transfer support,
pinned final support, sticky margin-0 support, and residual-norm-matched
dynamic top-k `2` evaluation rows. It also consumes the explicit
order-averaging probe, retention mitigation probe, update-decomposition audit,
value-mitigation gate, commutator-value-penalty probe, finite-update
order-control packet, and optional strategy review. The gate only selects
router-policy training if an intervention materially reduces commutator while
staying inside the CE guardrail; otherwise it records whether residual scale or
value composition is the more coherent next branch and keeps top-k `2`
causal-cooperation claims blocked. Add `--source-artifact-only` to skip the
fresh paired microtest and reproduce the older source-artifact interpretation.

After router-policy, simple value, low-rank value, commutator-aware value
penalty, and diagnostic order-averaging probes all exist, close out the
completed mitigation family without retraining:

```bash
python -m relaleap.experiments.promoted_topk2_post_value_router_mitigation_closeout \
  --out results/reports/token_larger_promoted_topk2_post_value_router_mitigation_closeout
```

This writes
`results/reports/token_larger_promoted_topk2_post_value_router_mitigation_closeout/summary.json`,
`source_rows.csv`, `closeout_rows.csv`, and `notes.md`. It fails closed if any
required mitigation packet is missing or inconsistent, keeps contextual top-k
`2` as the operational train-time support router, keeps order averaging
diagnostic-only, blocks top-k `2` causal-cooperation claims, and selects a
no-training pairwise value-interaction localization audit before proposing a
new mitigation family.

After the closeout selects pairwise value-interaction localization, run the
no-training source-artifact audit over the promoted top-k `2` causal-column
fingerprint packet:

```bash
python -m relaleap.experiments.promoted_topk2_pairwise_value_interaction_localization_audit \
  --out results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit
```

This writes
`results/reports/token_larger_promoted_topk2_pairwise_value_interaction_localization_audit/summary.json`,
`source_rows.csv`, `localization_rows.csv`, `column_localization_rows.csv`,
`stratum_rows.csv`, and `notes.md`. It consumes the per-token fixed-support
fingerprint rows, aggregate pair/column fingerprints, update-decomposition
audit, finite-update order-control report, and mitigation closeout. The report
localizes value-composition risk for mitigation design only; it keeps top-k `2`
causal-cooperation claims blocked and does not promote a new architecture.

If localization is diffuse or otherwise does not justify a new hub-family
mitigation, close the current top-k `2` value/router mitigation family and
select the next non-mitigation branch:

```bash
python -m relaleap.experiments.promoted_topk2_post_localization_closeout_report
```

This writes
`results/reports/token_larger_promoted_topk2_post_localization_closeout/summary.json`,
`source_rows.csv`, `closure_rows.csv`, and `notes.md`. It consumes the
post-value/router mitigation closeout, pairwise localization audit, active
top-k `1` next-evidence selector, active top-k `1` retention/functional-churn
follow-up, and optional strategy review. It fails closed on missing or
inconsistent source artifacts, records that no new top-k `2` value/router
mitigation family should be opened from diffuse localization evidence, and
selects the next matched deconfounding/retention branch without rerunning
training.

After the localization audit identifies a dominant hub-family value interaction,
run the bounded hub-focused value-composition mitigation probe:

```bash
python -m relaleap.experiments.promoted_topk2_hub_value_composition_mitigation_probe \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_promoted_topk2_hub_value_composition_mitigation_probe
```

This writes
`results/audits/token_larger_promoted_topk2_hub_value_composition_mitigation_probe/summary.json`,
`source_rows.csv`, `hub_value_composition_rows.csv`, `variant_metrics.csv`,
`phase_metrics.csv`, `per_token_commutator.csv`, and
`hub_value_composition_notes.md`. It adds opt-in local microtest variants that
penalize value-vector co-activation around the localized hub column, then gates
them against commutator reduction, transfer retention, support usage, CE drift,
and residual-stream L2. A passing local gate only selects a candidate for
RunPod validation; it does not promote top-k `2` causal-cooperation claims.

After refreshed per-token finite-update packets exist, turn the finite-update
evidence into an explicit causal-control matrix input without rerunning
training:

```bash
python -m relaleap.experiments.promoted_topk2_finite_update_order_control_audit
```

This writes
`results/reports/token_larger_promoted_topk2_finite_update_order_control_audit/summary.json`,
`variant_commutator.csv`, `per_token_commutator_strata.csv`, and
`causal_control_matrix_extension.csv`. The matrix extension covers promoted
contextual top-k-2, rank-matched contextual top-k-1, random fixed top-k-2, and
dense active-rank controls, while keeping top-k-2 causal-cooperation claims
blocked until downstream selection controls pass.

To materialize the explicit no-training finite-update control matrix from the
raw per-token commutator packet list, run:

```bash
python -m relaleap.experiments.promoted_topk2_finite_update_control_matrix
```

This writes
`results/reports/token_larger_promoted_topk2_finite_update_control_matrix/summary.json`,
`finite_update_control_matrix.csv`, `finite_update_control_strata.csv`,
`source_rows.csv`, and `notes.md`. It fails closed unless promoted contextual
top-k-2, rank-matched contextual top-k-1, random fixed top-k-2, and dense
active-rank controls all expose per-token forward-vs-reverse CE, symmetric KL,
logit MSE, support-set, token-position, support-churn, residual-norm, and
residual-delta fields. The report is a control-matrix input only, not causal
cooperation evidence by itself.

To join finite-update risk controls back to the deconfounded functional-benefit
strata before making any top-k `2` causal-cooperation claim, run:

```bash
python -m relaleap.experiments.promoted_topk2_finite_update_augmented_causal_gate
```

This writes
`results/reports/token_larger_promoted_topk2_finite_update_augmented_causal_gate/summary.json`,
`finite_update_risk_controls.csv`, `augmented_deconfounded_strata.csv`, and
`notes.md`. It consumes the deconfounded top-k `2` versus rank-matched top-k `1`
matched-strata audit plus the raw finite-update per-token commutator packet
list referenced by the control matrix. The gate keeps causal-cooperation claims
blocked unless functional benefit, fixed-support cleanliness, finite-update
logit-MSE risk, support churn, and random/dense controls all pass together.

To inspect the older fixed-singleton gain/regret packet that motivated the
singleton reconciliation audit, run:

```bash
python -m relaleap.experiments.active_topk1_singleton_gain_regret_diagnostic
```

This writes
`results/audits/token_larger_active_rank_matched_topk1_singleton_gain_regret_diagnostic/summary.json`,
`singleton_gain_by_context.csv`, `singleton_gain_by_stratum.csv`, and
`notes.md`. It consumes the source per-token causal-column fingerprint rows and
the deconfounded exact-context matching packet, compares raw, matched, and
unmatched context-level singleton gain/regret, and emits decision labels such as
`likely_real_singleton_gain_failure_mode`, `matching_artifact_possible`, and
`mixed_singleton_gain_evidence`.

To separate in-context router-selected singleton efficacy from off-context
forced-singleton stress in the same source artifact, run:

```bash
python -m relaleap.experiments.active_topk1_singleton_control_diagnostic
```

This writes
`results/audits/token_larger_active_rank_matched_topk1_singleton_control_diagnostic/summary.json`,
`singleton_control_by_context.csv`, `singleton_control_by_stratum.csv`, and
`notes.md`. It compares dominant-router singleton rows only where
`router_support_matches_fixed` is true against the best logged singleton on the
same context, records artifact hashes and the gain sign convention, and fails
closed when required source fields are missing. When the source causal-column
fingerprint has been refreshed after the singleton-control extension, it also
reports deterministic random singleton controls and the exhaustive same-context
singleton oracle denominator.

To reconcile the selected, logged-oracle, forced/off-context, random, and
exhaustive singleton controls under one gain sign convention, run:

```bash
python -m relaleap.experiments.active_topk1_singleton_reconciliation_audit
```

This writes
`results/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit/summary.json`,
`singleton_reconciliation_by_context.csv`,
`singleton_reconciliation_by_stratum.csv`, and `notes.md`. It treats
`singleton_gain = empty_loss - fixed_support_loss`, reports exact context
denominators for in-context router-selected singletons, same-context
logged-oracle singleton alternatives, forced/off-context dominant singleton
rows, deterministic random singleton controls, and the exhaustive same-context
singleton oracle. It still records random/exhaustive controls as missing when
the source artifact predates the singleton-control extension.

After the post-bracket research-direction report selects the context-conditioned
singleton interference decomposition, run the bounded local no-training audit:

```bash
python -m relaleap.experiments.active_topk1_context_conditioned_singleton_interference_audit
```

This writes
`results/audits/token_larger_active_topk1_context_conditioned_singleton_interference/summary.json`,
`singleton_interference_by_context.csv`, `singleton_interference_by_stratum.csv`,
`context_gate_holdout.csv`, and `notes.md`. It consumes the existing causal
fingerprint source rows and the post-bracket direction report, decomposes
selected own-context singleton gain from matched off-context forced-singleton
harm, includes matched top-k-2, random singleton, exhaustive singleton, and the
dense/rank-matched control provenance from the functional-retention packet, and
keeps broad reusable singleton claims excluded unless off-context interference
is cleanly gated. The current local packet passes with decision
`context_gate_reduces_offcontext_interference`; this is a local source-artifact
result rather than a backend-stable reusable singleton claim.

After the local decomposition packet exists, decide whether it is claim-changing
enough to spend one bounded backend validation run:

```bash
python -m relaleap.experiments.active_topk1_post_decomposition_decision_report
```

This writes
`results/reports/token_larger_active_topk1_post_decomposition_decision/summary.json`,
`decision_sources.csv`, and `notes.md`. It consumes the local decomposition
packet, the active top-k-1 backend provenance manifest, the functional-retention
audit, and the external strategy review when present. Under the current local
packet, the report recommends bounded RunPod validation when
`RELALEAP_GPU_BACKEND=runpod`, while still recording
`broad_reusable_singleton_claim_excluded`.

After the bounded backend closeout validates the context-conditioned singleton
decomposition, calibrate whether a deployable no-training context gate is good
enough to turn the diagnostic into a reusable singleton mechanism:

```bash
python -m relaleap.experiments.active_topk1_context_gate_suppression_calibration_audit
```

This writes
`results/audits/token_larger_active_topk1_context_gate_suppression_calibration/summary.json`,
`policy_metrics.csv`, `stratum_decisions.csv`, `bootstrap_intervals.csv`,
`source_rows.csv`, and `notes.md`. It consumes the context-conditioned
interference packet plus the backend closeout, then tests a deployable stratum
gate against pre-registered retained-gain, off-context harm-suppression, and
coverage-matched random-control criteria.

After the retention/churn bracket, singleton reconciliation, context-conditioned
interference, and deployable gate-calibration packets all exist, synthesize the
causal-retention interpretation without retraining:

```bash
python -m relaleap.experiments.active_topk1_causal_retention_synthesis_audit
```

This writes
`results/reports/token_larger_active_topk1_causal_retention_synthesis/summary.json`,
`source_rows.csv`, `evidence_rows.csv`, and `notes.md`. It records active
rank-matched top-k `1` as a local low-churn retention/control bracket with
context-gated singleton efficacy when supported, but fails closed on the broad
causal-retention claim unless the deployable context gate also passes its
suppression and random-control criteria.

If direct per-token pair synergy and the incremental matched top-k-2 gain gate
survive the deconfounded intervention audit, run the local null-controlled
synergy audit before making a causal-cooperation claim or spending Colab/GPU
cycles:

```bash
python -m relaleap.experiments.causal_synergy_null_audit
```

This writes
`results/audits/token_larger_topk2_causal_synergy_null_audit/summary.json`,
`matched_synergy_null_strata.csv`, and `notes.md`. It restricts the observed
top-k-2 pair-synergy estimate to the same matched strata as the deconfounded
audit, adds bootstrap uncertainty, compares against a stratified sign-flip null,
uses the available `fixed_best_support_swap` intervention as an artifact-level
matched control when present, and keeps `pair_synergy_supported` separate from
`cleaner_causal_bracket_supported`.

After refreshing the causal-column fingerprint artifact with matched control
pairs, run stronger anchored control checks before any Colab repeat or
causal-cooperation claim:

```bash
python -m relaleap.experiments.causal_synergy_null_audit \
  --control-intervention fixed_support_frequency_matched_control \
  --out results/audits/token_larger_topk2_causal_synergy_null_audit_support_frequency_control
python -m relaleap.experiments.causal_synergy_null_audit \
  --control-intervention fixed_random_nonrouter_control \
  --out results/audits/token_larger_topk2_causal_synergy_null_audit_random_nonrouter_control
python -m relaleap.experiments.causal_synergy_null_audit \
  --control-intervention fixed_loss_matched_nonrouter_control \
  --out results/audits/token_larger_topk2_causal_synergy_null_audit_loss_matched_control
```

This uses the same no-training matched-strata audit but compares the observed
router-selected top-k-2 pair synergy against fixed top-k-2 controls matched on
their observed anchor context. The current local artifact fails the
`fixed_best_support_swap`, support-frequency, random nonrouter, and loss-matched
control checks despite passing the sign-flip null, so broad top-k-2
causal-cooperation claims remain blocked.

To inspect whether that failure is local to a single artifact-level control or
holds per observed anchor across the sampled nonrouter controls in the same
causal fingerprint artifact, run:

```bash
python -m relaleap.experiments.causal_synergy_anchor_control_diagnostic
```

This writes
`results/audits/token_larger_topk2_causal_synergy_anchor_control_diagnostic/summary.json`,
`per_anchor_control_deltas.csv`, and `notes.md`. It pairs each sampled control
token row back to its observed router-selected anchor token row, reports
observed-minus-control pair-synergy distributions by anchor, and keeps
support-frequency, random nonrouter, and best-swap selection controls separate
from loss-matched outcome-proximal controls.

After the no-fallback support-frequency controls and per-anchor diagnostic are
refreshed, close or defer the broad top-k-2 causal-cooperation claim with the
command-driven stop report:

```bash
python -m relaleap.experiments.decision_report \
  --report topk2-causal-cooperation-stop
```

This writes
`results/reports/token_larger_topk2_causal_cooperation_stop_decision/decision_report.json`
and `.md`. The report keeps Colab top-k-2 replication deferred unless the local
sign-flip, selection-control, no-fallback matching, cleanliness, and per-anchor
checks all support the stronger claim.

After the local stop report closes the broad top-k-2 causal-cooperation claim,
select the next local causal-audit bracket and consume the support-frequency
candidate denominator artifact when present, without rerunning training:

```bash
python -m relaleap.experiments.decision_report \
  --report post-stop-causal-bracket-decision
```

This writes
`results/reports/token_larger_post_stop_causal_bracket_decision/decision_report.json`
and `.md`. The report combines the stop decision, the rank-matched contextual
top-k-1 bracket report, the existing top-k-2-vs-top-k-1 matched-strata audit,
and the exhaustive support-frequency candidate table when it exists. It selects
rank-matched contextual top-k-1 as the active local causal bracket and blocks a
support-frequency candidate-percentile claim when the exhaustive table is
missing or locally unidentified under the no-fallback support-count caliper.

To refresh that candidate denominator artifact without Colab/GPU replication,
rerun the causal-column fingerprint command locally:

```bash
python -m relaleap.experiments.causal_column_fingerprint \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_support_frequency_candidates \
  --load-balance-weights 0.0 \
  --max-pair-rows 8
```

This writes `support_frequency_candidate_controls.csv` alongside the existing
fingerprint artifacts. The table enumerates all nonrouter fixed-pair candidates
for each sampled router anchor, records exact/near support-count caliper
eligibility, marks unmatched candidates as excluded from the primary percentile
denominator instead of falling back loosely, and includes loss, singleton-gain,
residual-norm, random-rank, and pair-synergy fields for future percentile
audits. The current refreshed local artifact is present but has zero candidates
inside the declared support-count caliper, so the post-stop report classifies
the candidate-percentile bracket as artifact-ready but locally unidentified.

To diagnose that blocker without changing the claim gate, run the local
non-claim support-frequency blocker diagnostic:

```bash
python -m relaleap.experiments.support_frequency_blocker_diagnostic
```

This writes
`results/reports/token_larger_support_frequency_blocker_diagnostic/summary.json`,
`per_anchor_blockers.csv`, and `notes.md`. It consumes the existing
`support_frequency_candidate_controls.csv`, counts unmatched candidates by
failed caliper dimension, reports nearest-neighbor distance distributions, and
marks relaxed-caliper counts as exploratory rather than claim-bearing.
The post-stop causal bracket decision report consumes this diagnostic when it
exists, so the support-frequency candidate-percentile branch is explicitly
closed under the current no-fallback evidence instead of merely deferred as a
missing denominator.

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
After the contextual support-router promotion gate is satisfied, the
support-wide configs also use the contextual MLP support router with hidden dim
`128` by default.
To test whether the promoted contextual-router gain depends on full contextual
features or shallow shortcuts, run the feature-ablation support-head diagnostic:

```bash
python -m relaleap.experiments.contextual_router_shortcut_ablation \
  --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml \
  --out results/audits/token_larger_contextual_router_shortcut_ablation
```

The audit retrains the configured residual adapter, scores all token-larger
top-k `2` support pairs on the fixed in-repo batch, and trains hidden-only,
position-only, context-only, and full-context support heads against the fixed
support-loss matrix. It writes `summary.json`, `variant_metrics.csv`, and
`notes.md` with held-out oracle-target accuracy, fixed-support selector loss,
and realized support-intervention CE for each feature view.

After that shortcut ablation exists, choose the next non-duplicative promoted
top-k `2` diagnostic without rerunning training:

```bash
python -m relaleap.experiments.contextual_router_shortcut_decision_report
```

The report consumes the shortcut ablation plus the existing support-selection,
functional-churn, finite-update, simple value-mitigation, low-rank value, and
top-k `1` gate-suppression packets. It writes
`results/reports/token_larger_contextual_router_shortcut_decision/summary.json`,
`source_rows.csv`, `candidate_actions.csv`, and `notes.md`, keeps the shortcut
interpretation conservative, and emits one exact next command.

The selected commutator-aware value-penalty probe currently passes artifact
generation but does not establish a mitigation: the best penalty row reduces
anchor commutator logit MSE by only `0.23063087609231459`, below the `0.5`
gate.

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
`router_target_contextual_diagnostic.csv`,
`router_target_contextual_sequence_diagnostic.csv`,
`router_support_intervention.csv`, `router_support_sequence_intervention.csv`,
`contextual_router_support_head.csv`,
`contextual_router_support_sequence_head.csv`, and `notes.md`. The summary
reports per-token oracle-support regret, best global fixed support, dominant
router support, one-swap recovery, support-loss distribution, pairwise synergy
leaders, the existing router support audit, and linear, small-MLP, and
contextual small-MLP router-target probes. The legacy probes train against the
per-token oracle support pair on even flattened token positions and report
holdout odd-position recovery of the oracle gap. The sequence diagnostics train
on even full sequences and hold out odd full sequences for the contextual
oracle-target probe and train-time-style contextual support head. Intervention
CSVs include split-local router loss, oracle loss, intervention loss, absolute
intervention-minus-router loss, oracle regret, and recovery fraction so tiny
oracle-gap denominators can be interpreted conservatively. The contextual
probes add normalized token position plus immediate previous/next
hidden-neighborhood features. The decision report fails closed on missing audit
artifacts and selects whether the next branch should target router support
selection, column redundancy, or pairwise composition.

To check whether the promoted token-larger contextual top-k-2 support-selection
packet already satisfies the stricter sequence-level holdout coverage requested
by strategic review, run:

```bash
python -m relaleap.experiments.promoted_topk2_sequence_holdout_coverage_report
```

This no-training report writes
`results/reports/token_larger_promoted_topk2_sequence_holdout_coverage/summary.json`,
`split_rows.csv`, `source_rows.csv`, and `notes.md`. It consumes the promoted
support-selection quality audit, exhaustive support audit, causal-adequacy
matrix, and optional external strategy review. It fails closed on missing source
artifacts and records whether the existing support-prediction holdout is only an
even/odd flattened-token-position split or includes a sequence-level split. When
sequence coverage exists but the sequence-heldout contextual support head is
worse than the learned router, the report emits
`sequence_holdout_support_head_generalization_failed`, blocks deployable
contextual support-selection claims, and selects a local K-fold
sequence-heldout causal-feature ablation of the actual promoted contextual
router versus linear/top-k controls.

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
