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
- Control synergy mean: `0.22994570826485777`
- Observed-minus-control synergy mean: `-0.06117739653267449`
- Observed-minus-control CI: `[-0.10331990785751402, -0.01591229749678766]`
- Control match diagnostics: `{'support_count_difference_mean': -10.705466087448437, 'support_count_difference_abs_mean': 10.705466087448437, 'fixed_support_loss_difference_mean': -0.09749224139936252, 'fixed_support_loss_difference_abs_mean': 0.09749224139936252, 'matched_anchor_support_count': 99, 'matched_control_support_count': 99, 'control_exact_support_count_candidate_count_mean': 1.792285354850137, 'control_near_support_count_candidate_count_mean': 2.8864836940604213, 'control_min_support_count_abs_difference_available_mean': 0.0, 'control_match_status_counts': {'exact_support_count_candidate_available': 790}, 'matched_strata_overlap_fraction': 1.0}`
- Pair synergy supported: `False`
- Cleaner causal bracket supported: `False`
- Control note: `fixed_best_support_swap` is an artifact-level matched control, not a retraining control.
