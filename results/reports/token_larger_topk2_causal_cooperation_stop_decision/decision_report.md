# Top-k-2 Causal Cooperation Stop Decision

- Status: `pass`
- Decision: `stop_topk2_causal_cooperation_claim`
- Top-k-2 causal cooperation claim closed locally: `True`
- Colab top-k-2 replication warranted: `False`
- Keep contextual router default: `True`
- Future causal-audit bracket: `rank_matched_topk1_or_support_frequency_candidate_percentile`

## Rationale

The visible top-k-2 pair synergy survives only the weak sign-flip null. The best-swap selection control is negative, top-k-2 misses the fixed-support and functional-churn cleanliness gates, and the support-frequency-calipered no-fallback controls are unidentified rather than supportive. This locally closes the broad top-k-2 causal-cooperation claim while keeping the promoted contextual router as an empirical support-routing default.

## Metrics

| Metric | Value |
| --- | --- |
| `observed_deconfounded_pair_synergy_mean` | `0.16724165783875575` |
| `observed_deconfounded_pair_synergy_ci` | `[0.13490017275549018, 0.1993658205307805]` |
| `best_swap_observed_minus_control_synergy_mean` | `-0.06270404575864595` |
| `best_swap_observed_minus_control_synergy_ci` | `[-0.10542973940582488, -0.016833687727758676]` |
| `topk2_fixed_support_cleaner_strata_fraction` | `0.7037037037037037` |
| `topk2_functional_churn_cleaner_strata_fraction` | `0.7407407407407407` |
| `cleaner_fraction_threshold` | `0.8` |
| `no_fallback_control_matched_strata_counts` | `{'fixed_support_frequency_matched_control': 0, 'fixed_random_nonrouter_control': 0, 'fixed_loss_matched_nonrouter_control': 0, 'fixed_singleton_gain_matched_nonrouter_control': 0, 'fixed_residual_norm_matched_nonrouter_control': 0}` |

## Signals

- sign_flip_synergy_supported: `True`
- best_swap_rejects_topk2: `True`
- no_fallback_controls_unidentified: `True`
- no_fallback_anchor_controls_unidentified: `True`
- cleaner_gates_fail: `True`
- anchor_rejects_claim: `True`

## Next Step

use rank-matched contextual top-k-1 or a support-frequency candidate-percentile bracket for the next local causal audit; do not run Colab top-k-2 replication for the closed broad claim
