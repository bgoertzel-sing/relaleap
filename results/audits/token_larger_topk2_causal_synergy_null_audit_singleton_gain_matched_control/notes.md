# Causal Synergy Null Audit

- Status: `pass`
- Decision: `pair_synergy_requires_artifact_random_pair_controls`
- Source audit: `results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1`
- Deconfounded audit: `results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention`
- Observed matched strata: `28`
- Observed synergy mean: `0.16724165783875575`
- Observed synergy CI: `[0.13490017275549018, 0.1993658205307805]`
- Sign-flip null mean: `0.00022764360724959428`
- Sign-flip one-sided p-value: `0.000999000999000999`
- Control available: `False`
- Control synergy mean: `None`
- Observed-minus-control synergy mean: `None`
- Observed-minus-control CI: `[None, None]`
- Control match diagnostics: `{'support_count_difference_mean': None, 'support_count_difference_abs_mean': None, 'fixed_support_loss_difference_mean': None, 'fixed_support_loss_difference_abs_mean': None, 'matched_anchor_support_count': 0, 'matched_control_support_count': 0, 'control_exact_support_count_candidate_count_mean': None, 'control_near_support_count_candidate_count_mean': None, 'control_min_support_count_abs_difference_available_mean': None, 'control_match_status_counts': {}, 'matched_strata_overlap_fraction': 0.0}`
- Pair synergy supported: `False`
- Cleaner causal bracket supported: `False`
- Control note: `fixed_singleton_gain_matched_nonrouter_control` is an artifact-level matched control, not a retraining control.
