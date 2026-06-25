# Deconfounded Intervention Audit

- Status: `pass`
- Decision: `topk2_pair_synergy_survives_deconfounding_but_cleanliness_bar_fails`
- Source audit: `results/audits/token_larger_support_wide_promoted_default_causal_column_fingerprint_stability_topk1`
- CE guardrail tolerance: `0.05`
- Matched deconfounded strata: `54`
- Top-k-2 alpha-0 CE: `2.9124014377593994`
- Rank-matched top-k-1 alpha-0 CE: `2.8664543628692627`
- Top-k-2 CE deficit: `0.04594707489013672`
- CE guardrail passed: `True`
- Top-k-2 fixed delta minus top-k-1 mean: `-0.22483390948866877`
- Top-k-2 fixed-support cleaner strata fraction: `0.7037037037037037`
- Top-k-2 logit MSE minus top-k-1 mean: `-0.03353773377860789`
- Top-k-2 functional-churn cleaner strata fraction: `0.7407407407407407`
- Coarse top-k-2 pair synergy mean: `0.18841657638549805`
- Per-token pair synergy available: `True`
- Deconfounded top-k-2 pair synergy mean: `0.18339446802263767`
- Deconfounded top-k-2 pair synergy positive strata fraction: `0.9629629629629629`
- Active-rank note: active_rank_proxy is reported as a bracket dimension, not exact-matched, because promoted top-k-2 and rank-matched top-k-1 have structurally different active-rank proxies in this artifact.
