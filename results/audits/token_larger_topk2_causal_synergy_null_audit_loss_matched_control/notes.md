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
- Control synergy mean: `0.1976252286511404`
- Observed-minus-control synergy mean: `-0.028856916918957035`
- Observed-minus-control CI: `[-0.06367245141068885, 0.006157264175201656]`
- Control match diagnostics: `{'support_count_difference_mean': -14.38798343172024, 'support_count_difference_abs_mean': 14.38798343172024, 'fixed_support_loss_difference_mean': 0.02460347855153629, 'fixed_support_loss_difference_abs_mean': 0.026243893255734525, 'matched_anchor_support_count': 110, 'matched_control_support_count': 238, 'control_exact_support_count_candidate_count_mean': 0.0, 'control_near_support_count_candidate_count_mean': 0.0, 'control_min_support_count_abs_difference_available_mean': 14.38798343172024, 'control_match_status_counts': {'loose_support_count_fallback': 1891}, 'matched_strata_overlap_fraction': 1.0}`
- Pair synergy supported: `False`
- Cleaner causal bracket supported: `False`
- Control note: `fixed_loss_matched_nonrouter_control` is an artifact-level matched control, not a retraining control.
