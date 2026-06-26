# Promoted Top-k-2 Update Decomposition Audit

- Status: `pass`
- Decision: `value_update_dominated_order_sensitivity`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml`
- Full anchor commutator logit MSE: `0.21909268200397491`
- Router-only fraction of full: `0.22705120722820915`
- Value-only fraction of full: `1.2560724600135986`

## Rationale

Value-only transfer updates preserve a material share of the full commutator signal while router-only transfer updates fall below the `0.5` fraction threshold.

## Next Step

test value-update regularization or lower-rank value updates before changing router policy
