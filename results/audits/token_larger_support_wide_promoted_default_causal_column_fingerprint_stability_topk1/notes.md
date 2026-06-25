# Causal Column Fingerprint Audit

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_causal_column_fingerprint`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml`
- Status: `ok`
- Variants: `baseline, load_balance_0.0125, rank_matched_topk1_contextual`
- Column fingerprint rows: `96`
- Pair intervention rows: `1200`
- Per-token pair intervention rows: `60480`
- Pair synergy sign convention: pair_synergy = pair_gain - singleton_left_gain - singleton_right_gain, where gain = empty_loss - intervention_loss; positive values mean the fixed pair improves loss more than the sum of its singleton interventions under this loss-space diagnostic.

## Variant Summary

- `baseline`: alpha-0 CE `2.9124014377593994`, used columns `19`, unique support sets `41`, load entropy `2.523192169717511`, mean abs ablate delta `0.05213449398676554`, mean abs force delta `1.321441113948822`
- `load_balance_0.0125`: alpha-0 CE `2.830270528793335`, used columns `24`, unique support sets `45`, load entropy `2.887798705299062`, mean abs ablate delta `0.05555599927902222`, mean abs force delta `1.413821558157603`
- `rank_matched_topk1_contextual`: alpha-0 CE `2.8664543628692627`, used columns `31`, unique support sets `31`, load entropy `3.264069075886517`, mean abs ablate delta `0.027024442950884502`, mean abs force delta `1.3583199083805084`
