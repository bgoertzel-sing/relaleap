# Finite-update-augmented causal gate

Status: `pass`
Decision: `finite_update_augmented_topk2_causal_cooperation_blocked`

This no-training gate joins deconfounded functional-benefit strata to finite-update order-sensitivity controls. It is a claim gate, not a new training result.

Claim gate: `causal_cooperation_blocked_unless_functional_benefit_survives_finite_update_risk_controls`
Augmented strata: `46`
Positive benefit fraction: `0.6521739130434783`
Fixed-support cleaner fraction: `0.6521739130434783`
Top-k-2 minus top-k-1 finite logit MSE: `0.2784630590280149`
Top-k-2 finite support churn: `0.9247085715595466`
Next step: keep top-k-2 causal-cooperation claims blocked; return to active rank-matched top-k-1 controls, retention/churn, or matched deconfounding
