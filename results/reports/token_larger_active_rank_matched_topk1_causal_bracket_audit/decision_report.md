# Active Rank-Matched Top-k-1 Causal Bracket Audit

- Status: `pass`
- Decision: `confirm_active_rank_matched_topk1_causal_bracket`
- Rank-matched top-k-1 primary causal bracket: `True`
- Top-k-2 reference condition only: `True`
- Top-k-2 causal cooperation claim supported: `False`
- Support-frequency percentile claim supported: `False`
- Colab replication warranted: `False`

## Rationale

The exact-context deconfounded audit is complete and supports using rank-matched contextual top-k-1 as the primary local causal bracket. Top-k-2 stays inside the CE guardrail only narrowly and remains useful as a reference condition, but it misses the incremental-gain, fixed-support cleanliness, and functional-churn gates needed for a comparative causal-cooperation claim.

## Evidence

- Audit: `results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention`
- Source decision: `topk2_comparative_causal_cooperation_not_supported`

## Metrics

| Metric | Value |
| --- | ---: |
| `topk1_alpha0_ce_loss` | 2.86645436 |
| `topk2_alpha0_ce_loss` | 2.91240096 |
| `topk2_ce_deficit_vs_topk1` | 0.04594660 |
| `ce_guardrail_tolerance` | 0.05000000 |
| `matched_exact_context_count` | 210.00000000 |
| `matched_topk1_context_fraction` | 0.83333333 |
| `matched_topk2_context_fraction` | 0.83333333 |
| `unmatched_topk1_context_count` | 42.00000000 |
| `unmatched_topk2_context_count` | 42.00000000 |
| `deconfounded_topk2_pair_synergy_mean` | 0.21385600 |
| `topk2_incremental_pair_gain_minus_topk1_singleton_mean` | 0.02732963 |
| `topk2_incremental_pair_gain_positive_strata_fraction` | 0.65217391 |
| `topk2_fixed_support_cleaner_strata_fraction` | 0.65217391 |
| `topk2_functional_churn_cleaner_strata_fraction` | 0.60869565 |
| `claim_fraction_threshold` | 0.80000000 |

## Signals

- topk1_ce_primary: `True`
- exact_context_coverage_present: `True`
- topk2_reference_within_guardrail: `True`
- topk2_comparative_gates_fail: `True`

## Next Step

run a bounded local causal-separability audit for the active rank-matched contextual top-k-1 bracket, with top-k-2 retained only as a reference and support-frequency percentile claims closed
