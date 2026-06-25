# Causal Audit Coverage Report

- Status: `pass`
- Decision: `rank_matched_topk1_active_post_stop_bracket`
- Source artifacts: `10`
- Row granularity: `per_token`
- Matched strata count: `5`
- Top-k-2 CE deficit vs rank-matched top-k-1: `0.045946598052978516`
- Missing fields: `[]`
- Missing controls: `[]`
- Post-stop rank-matched top-k-1 active: `True`
- Support-frequency candidate-percentile identified: `False`

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

- Active bracket: `rank_matched_contextual_topk1`
- Blocked claims: `{'topk2_causal_cooperation': True, 'support_frequency_candidate_percentile': True}`
- Match/bin by: `position_bin, token_class, support_frequency_or_dominance, residual_norm_or_gain_bin, active_rank_proxy`

## Next Step

use rank-matched contextual top-k-1 as the active local causal bracket and keep top-k-2 causal-cooperation and support-frequency percentile claims blocked
