# Causal Synergy Null Audit

- Status: `pass`
- Decision: `pair_synergy_not_supported_against_local_control_null`
- Source audit: `results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1`
- Deconfounded audit: `results/audits/token_larger_topk2_vs_rank_matched_topk1_deconfounded_intervention`
- Observed matched strata: `28`
- Observed synergy mean: `0.16876831173218335`
- Observed synergy CI: `[0.13735732606108536, 0.2003492825608546]`
- Sign-flip null mean: `0.00021014414197889728`
- Sign-flip one-sided p-value: `0.000999000999000999`
- Control available: `True`
- Control synergy mean: `0.21449617696727366`
- Observed-minus-control synergy mean: `-0.04572786523509039`
- Observed-minus-control CI: `[-0.07762627643435849, -0.013758558338598462]`
- Control match diagnostics: `{'support_count_difference_mean': -14.400817220171778, 'support_count_difference_abs_mean': 14.400817220171778, 'fixed_support_loss_difference_mean': 0.08039094616849454, 'fixed_support_loss_difference_abs_mean': 0.08059168414090194, 'matched_anchor_support_count': 110, 'matched_control_support_count': 246, 'control_exact_support_count_candidate_count_mean': 0.0, 'control_near_support_count_candidate_count_mean': 0.0, 'control_min_support_count_abs_difference_available_mean': 14.400817220171778, 'control_match_status_counts': {'loose_support_count_fallback': 1812}, 'matched_strata_overlap_fraction': 1.0}`
- Pair synergy supported: `False`
- Cleaner causal bracket supported: `False`
- Control note: `fixed_residual_norm_matched_nonrouter_control` is an artifact-level matched control, not a retraining control.
