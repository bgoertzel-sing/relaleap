# Causal Audit Coverage Report

- Status: `pass`
- Decision: `existing_artifacts_sufficient_for_next_no_training_audit`
- Source artifacts: `9`
- Row granularity: `per_token`
- Matched strata count: `5`
- Top-k-2 CE deficit vs rank-matched top-k-1: `0.04594683647155762`
- Missing fields: `[]`
- Missing controls: `[]`

## Controls

- promoted_topk2: `True`
- rank_matched_topk1: `True`
- random_support: `True`
- norm_matched_dense: `True`
- retention_dense_control: `True`

## Matching Fields

- position_bin: `True`
- token_class: `True`
- support_frequency: `True`
- residual_norm_or_gain: `True`
- residual_norm_bin: `True`
- active_rank_proxy: `True`
- per_token_rows: `True`

## Next Matrix

- Brackets: `promoted_contextual_topk2, rank_matched_contextual_topk1`
- Match/bin by: `position_bin, token_class, support_frequency_or_dominance, residual_norm_or_gain_bin, active_rank_proxy`

## Next Step

run the no-training residual-norm/active-rank/support-stratum matched top-k-2 versus rank-matched top-k-1 causal audit
