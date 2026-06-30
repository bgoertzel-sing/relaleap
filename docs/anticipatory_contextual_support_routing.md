# Anticipatory Contextual Support Routing

## Motivation

Ben's 2026-06-27 direction is to prioritize an anticipatory contextual support
routing branch. The hypothesis is that the earlier full-context contextual MLP
router was useful because it had access to future-context features, but that
direct use is nondeployable. A deployable router might recover some of the same
signal by predicting those future-context features from allowed causal inputs,
then routing columns from the predicted features.

This branch should be treated as a targeted mechanism test, not a generic
router-size increase and not a return to CE-only promotion.

## Non-cheating definition

The router is non-cheating only if, at inference time, it uses features
available at or before the current token/position:

- current hidden state;
- past hidden states or a causal sequence summary;
- current/past support choices;
- current/past residual norms, logit entropy, or support confidence;
- current position or causal position encoding;
- learned predictions of future-context features generated from those causal
  inputs.

The predictor may be trained against future-context targets extracted from the
full-context oracle router, but the future-context target vectors must not be
read directly by the router during evaluation.

## 2026-06-30 priority: Transformer-ACSR

Ben's follow-up direction is to prioritize a smarter ACSR predictor rather than
treat the earlier shallow MLP/GRU result as decisive. The promoted
`contextual_mlp` top-k2 router uses full-context `next_hidden` and
`next_delta` chunks, so it should be treated as a nondeployable teacher. The
next ACSR branch should ask whether a strictly causal sequence model can
predict enough of those teacher chunks to recover useful support routing
without reading future hidden states directly.

The preferred next bounded experiment is `transformer_acsr`: a small causal
transformer over prefix-safe features that predicts future-router feature
chunks and/or teacher router logits from causal inputs only, then routes
columns using the predicted chunks. This supersedes the previous instruction to
start with another MLP/GRU-only ACSR pass. MLP/GRU predictors should remain
controls or ablations, not the main branch.

## Concrete experiment

Implement a command-driven pilot tentatively named
`transformer_acsr`.

The pilot should:

1. Reuse the promoted token-larger support-wide setting as the first local
   scale.
2. Extract the full-context oracle feature vector used by the nondeployable
   contextual MLP router.
3. Train a small causal transformer to reconstruct that future-context feature
   vector from causal inputs only.
4. Feed the predicted future-context features, not the true future features,
   into a support router.
5. Compare against the existing controls:
   - promoted full-context top-k2 contextual router;
   - causal-feature-safe contextual top-k2 router;
   - prior MLP/GRU ACSR predictors where available;
   - linear top-k2 router;
   - rank-matched contextual top-k1;
   - random/fixed top-k2;
   - norm-matched dense active-rank control where already available.

Preferred predictor/target design:

- small causal transformer over token prefix, current/previous hidden, causal
  position encoding, recent support choices, residual norms, entropy/confidence,
  and any other prefix-safe sequence summaries;
- multi-target training against `next_hidden`, `next_delta`, full-context
  router logits, and/or softened top-k support distribution;
- optional uncertainty/calibration head so low-confidence predicted future
  chunks can be damped rather than over-routed;
- MLP, GRU, token/position-only transformer, and shuffled/misaligned target
  predictors as controls.

## Required controls

The branch must include nulls that distinguish real anticipatory signal from
cheap CE or position shortcuts:

- shuffled predicted-feature control;
- delayed or misaligned predicted-feature control;
- token/position-only predicted-feature control;
- same-parameter MLP/GRU predictor controls;
- same-student support intervention: predicted-feature support versus
  token/position-null support through the same trained residual values;
- causal-feature perturbation check showing that future positions do not change
  router outputs;
- full-context `contextual_mlp` teacher as an explicit oracle/nondeployable
  upper bound, not as deployable evidence;
- retention/churn evaluation after A-to-B adaptation, not just fixed-batch CE.

## Promotion criteria

Do not promote the branch from CE alone. ACSR should only be treated as a real
mechanism improvement if it satisfies a combined gate:

- improves or matches CE against causal-feature-safe top-k2;
- closes part of the full-context oracle gap without reading future features;
- does not worsen oracle-support regret versus causal-feature-safe top-k2;
- lowers functional churn or support churn versus promoted full-context top-k2;
- beats shuffled and token/position-only predicted-feature controls;
- does not underperform rank-matched top-k1 on the retention/churn guardrail
  unless it gives a clearly compensating CE/support-regret gain.

If it improves CE but worsens support regret/churn, record it as another
operational CE router rather than as causal column-selection evidence.

## First bounded automation step

The next RelaLeap automation step should create a fail-closed design report for
Transformer-ACSR before any GPU work. The report should identify:

- exact full-context feature tensors to predict;
- exact causal input tensors;
- transformer size, causal mask, context window, and local CPU smoke budget;
- MLP/GRU and token/position-only control predictors;
- first local config and output directory;
- artifact schema;
- controls and nulls;
- pass/fail criteria;
- the smallest implementation/test patch needed for a local smoke run.

After that design report passes, implement the local pilot and run only local
CPU-scale evidence until the CE plus non-CE gates are discriminative.
