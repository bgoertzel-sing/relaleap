# Promoted Top-k-2 Post-Value/Router Mitigation Closeout

- Status: `pass`
- Decision: `promoted_topk2_mitigation_closeout_no_promotion`
- Selected next action: `pairwise_value_interaction_localization_audit`
- Next command: `None`

## Evidence

- Router-policy reduction fraction: `-0.013875435318412323`
- Simple value best reduction fraction: `0.1266509238864246`
- Low-rank value best reduction fraction: `0.08746147951831357`
- Commutator value-penalty best reduction fraction: `0.23063087609231459`
- Order-averaging logit-MSE ratio: `0.24796639122347383`
- Value-only/full commutator fraction: `1.2560724600135986`

## Rationale

Router-policy pinning/freezing did not reduce the commutator, simple value clipping/scaling did not clear the gate, low-rank value updates did not clear the gate, and commutator-aware value penalties did not clear the gate. Explicit order averaging remains diagnostic-only. Because the update decomposition is value-dominated, the next non-duplicative step is to localize pairwise fixed-value interactions before designing another trainable mitigation.

## Next Step

implement a no-training pairwise value-interaction localization audit over the promoted top-k-2 checkpoint before proposing another mitigation family
