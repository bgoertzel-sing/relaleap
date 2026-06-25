# Active Top-k-1 Functional-Retention Audit

- Status: `pass`
- Decision: `functional_retention_bracket_only`
- Claim status: `blocked_by_negative_singleton_gain`
- Mean top-k-1 support churn: `0.005859375`
- Mean top-k-2 support churn: `0.86328125`
- Minimum support-churn advantage: `0.8046875`
- Minimum logit-churn advantage: `0.011958405375480652`
- Mean source singleton gain: `-0.042048803633778095`

## Rationale

The active rank-matched contextual top-k-1 bracket remains useful as a low-churn functional-retention bracket, but the current packets do not support a singleton causal-retention claim. The source singleton gain is still negative, and the current microtest records A-to-B transfer retention rather than a finite-update A-to-B versus B-to-A commutator.

## Evidence Separation

- Support identity churn: exact support-set churn from the completed probe packets.
- Functional/logit churn: anchor logit MSE drift after transfer.
- Causal gain/regret: source singleton gain remains the causal-gain caveat.
- CE guardrail: positive anchor CE deterioration must stay within `0.05`.
- Finite-update commutator: missing from current packets; no A-to-B/B-to-A claim is made.

## Next Step

extend the local microtest to include finite-update order sensitivity A-to-B versus B-to-A before spending Colab time on a stronger causal retention claim
