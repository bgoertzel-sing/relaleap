# Active Top-k-1 Retention/Functional-Churn Follow-up

- Status: `pass`
- Decision: `retention_functional_churn_bracket_supported`
- Claim status: `local_retention_functional_churn_bracket_only`
- Mean top-k-1 support churn: `0.005859375`
- Mean top-k-2 support churn: `0.86328125`
- Mean random fixed top-k-2 support churn: `0.0`
- Mean top-k-1 transfer advantage vs top-k-2: `0.0297243595123291`
- Mean top-k-1 transfer advantage vs dense: `0.47648465633392334`
- Mean top-k-1 transfer advantage vs random fixed top-k-2: `0.30895793437957764`
- Minimum top-k-1 commutator advantage vs controls: `0.04532540775835514`

## Rationale

The selected retention/functional-churn branch is supported locally across the refreshed probe packets. Rank-matched contextual top-k-1 has much lower support-identity churn than promoted top-k-2, no higher functional/logit churn, lower finite-update commutator risk, and better transfer CE improvement than promoted top-k-2, dense active-rank, and random fixed top-k-2 controls. This remains a retention/churn bracket, not a renewed top-k-2 causal-cooperation claim.

## Next Step

run a backend-stable RunPod repeat of the retention/functional-churn follow-up only if the next causal-retention claim needs GPU parity; otherwise use this local bracket to design the next discriminative causal-retention audit
