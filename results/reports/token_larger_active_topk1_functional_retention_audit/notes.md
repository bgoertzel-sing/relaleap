# Active Top-k-1 Functional-Retention Audit

- Status: `pass`
- Decision: `functional_retention_bracket_only`
- Claim status: `blocked_by_negative_singleton_gain`
- Mean top-k-1 support churn: `0.005859375`
- Mean top-k-2 support churn: `0.85546875`
- Minimum support-churn advantage: `0.8046875`
- Minimum logit-churn advantage: `0.011935070157051086`
- Minimum commutator anchor-logit advantage: `0.19262094981968403`
- Minimum commutator transfer-logit advantage: `0.19381753914058208`
- Mean source singleton gain: `-0.042048803633778095`

## Rationale

The active rank-matched contextual top-k-1 bracket remains useful as a low-churn functional-retention bracket, but the current packets do not support a singleton causal-retention claim. The source singleton gain is still negative, so any finite-update order-sensitivity advantage remains bracket evidence rather than a causal-retention claim.

## Evidence Separation

- Support identity churn: exact support-set churn from the completed probe packets.
- Functional/logit churn: anchor logit MSE drift after transfer.
- Causal gain/regret: source singleton gain remains the causal-gain caveat.
- CE guardrail: positive anchor CE deterioration must stay within `0.05`.
- Finite-update commutator: A-to-B versus B-to-A final-function logit MSE when present.

## Next Step

use the finite-update order-sensitivity evidence to decide whether a targeted Colab/GPU repeat is worth running despite the negative singleton-gain blocker
