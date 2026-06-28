# Core/Periphery Predictive-Coding Columns For ACSR

Date: 2026-06-28.

## Motivation

Ben's 2026-06-28 direction is to return to the original HIBACAML-style idea
behind RelaLeap: columns should not only be routed well; they should also have
interesting internal learning structure. The current ACSR work improved support
discovery and produced useful diagnostics, but the column internals remained
close to ordinary backprop-trained residual atoms. That leaves the central
continual-learning hypothesis under-tested.

The new provisional hypothesis is:

```text
ACSR should handle anticipatory routing among columns, while each selected
column should learn through a core/periphery predictive-coding organization
that protects task-generic reusable structure in the core and leaves more
task-specific, pruneable, causally fragile structure in the periphery.
```

This direction does not promote the existing sparse ACSR branch. It uses the
negative ACSR evidence as a diagnosis: routing alone is not enough.

## Current Baseline To Move Beyond

The current residual columns are mostly standard differentiable residual
adapters:

- a frozen base transformer emits hidden states;
- a router selects top-k columns;
- selected columns mix learned atom values;
- the residual adapter adds the selected values to the hidden state;
- router, atom logits, and atom values are trained by ordinary backpropagation,
  usually supervised next-token CE;
- PC-style losses and HEP settling exist as probes, but are not the default
  internal column-learning mechanism.

This setup tests sparse support selection more than HIBACAML-style column
organization.

## Proposed Column Structure

Each residual column should be split into at least two internal rings or
submodules:

- **core ring**: lower-plasticity, task-generic, protected units/atoms intended
  to encode reusable predictive structure;
- **peripheral ring**: higher-plasticity, context/task-specific units/atoms
  intended to absorb recent adaptation and be preferentially pruned when causal
  intervention evidence says they are non-essential or harmful.

The router still chooses columns, but the selected column emits:

```text
column_residual =
  core_gate * core_prediction_correction
  + periphery_gate * peripheral_prediction_correction
```

The core/periphery gates may depend on causal ACSR features, prediction
uncertainty, recent support stability, and column-specific genericity scores,
but they may not use future labels or future hidden states at evaluation time.

## Predictive-Coding Learning Proposal

Within a selected column, train local predictive modules rather than only
direct residual values:

- predict the base hidden-state delta or teacher residual target from causal
  hidden/context features;
- compute local prediction errors for the core and periphery;
- update core parameters slowly when errors recur across contexts/tasks;
- update periphery parameters quickly when errors are context-specific;
- penalize periphery reliance when the core can explain the same correction;
- protect core weights during later task adaptation unless causal intervention
  shows they are task-specific or harmful.

A first implementable objective can be a hybrid:

```text
loss =
  supervised_CE_guardrail
  + lambda_pc * local_hidden_or_teacher_residual_prediction_error
  + lambda_core_genericity * core_cross_context_consistency
  + lambda_periphery_sparsity * peripheral_usage_or_norm
  + lambda_retention * anchor_context_drift
```

The CE term remains a guardrail, not the primary scientific claim.

## Causal Pruning Hypothesis

Causal pruning should preferentially remove peripheral atoms/units before core
ones. A pruning candidate is more peripheral if it has:

- low cross-context support or transfer;
- low retention benefit on anchor contexts;
- high functional churn;
- high off-target intervention effect;
- weak or inconsistent causal necessity/sufficiency fingerprints;
- high task specificity under task-free continual-learning slices.

Core units should be retained when they show broad causal utility, low
off-target drift, low commutator contribution, and useful transfer across
contexts.

## ACSR Interaction

ACSR remains scientifically useful as a router/diagnostic shell:

- it predicts future-context features from causal inputs;
- it chooses which columns should be active;
- it can provide prediction uncertainty and support-margin signals to the
  selected columns;
- it should be evaluated against shuffled-feature, token/position-only,
  same-student, dense/rank/norm, and MLP controls.

But ACSR should not be treated as sufficient. The new question is whether ACSR
plus core/periphery PC columns beats both current sparse ACSR and dense/MLP
controls on non-CE objectives.

## First Bounded Gate

Do not start with a large GPU run. The first automation step should be a local
design/contract gate named something like:

```text
core_periphery_pc_column_design
```

It should specify:

- exact core and periphery parameterization;
- local prediction target: hidden delta, teacher residual, or both;
- update/plasticity rules for core versus periphery;
- consolidation/genericity score;
- periphery-first pruning criterion;
- ACSR features available to the column;
- matched controls: current sparse ACSR, dense rank/norm, parameter-matched
  causal MLP, random/frequency support, and no-core/no-periphery ablations;
- artifact schema for retention, churn, intervention fingerprints,
  commutators, residual-L2, and CE guardrails.

The first code-bearing run should be a tiny local pilot only after the design
gate has no open schema/mechanism gaps.

## Promotion Criteria

This direction should advance only if the core/periphery PC column shows
evidence beyond CE:

- lower task-free continual-learning forgetting than current sparse ACSR and
  dense/MLP controls at matched residual norm or active compute;
- lower finite-update commutator or order sensitivity;
- cleaner causal intervention fingerprints;
- successful periphery-first pruning with minimal anchor-task drift;
- useful target/context adaptation without excessive functional churn;
- CE/perplexity not worse beyond a preregistered guardrail.

If it improves CE but does not improve retention, commutator, pruning, or
causal fingerprints, treat it as an operational adapter variant, not evidence
for the HIBACAML-style column hypothesis.

## Relationship To The Latest ACSR Closeout

The latest ACSR negative-evidence closeout demoted the existing sparse ACSR
promotion path and selected dense/MLP controls as mechanism assays. This new
Ben direction should be interpreted as a higher-level research correction:
dense/MLP controls remain mandatory, but the next sparse-column hypothesis
should modify the within-column learning architecture, not merely rerun ACSR
support routing.
