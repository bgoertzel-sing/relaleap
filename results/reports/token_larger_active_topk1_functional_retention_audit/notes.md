# Active Top-k-1 Functional-Retention Audit

- Status: `pass`
- Decision: `functional_retention_bracket_only`
- Claim status: `context_gated_singleton_efficacy_with_offcontext_interference`
- Mean top-k-1 support churn: `0.005859375`
- Mean top-k-2 support churn: `0.85546875`
- Minimum support-churn advantage: `0.8046875`
- Minimum logit-churn advantage: `0.011935070157051086`
- Minimum commutator anchor-logit advantage: `0.19262094981968403`
- Minimum commutator transfer-logit advantage: `0.19381753914058208`
- Mean source singleton gain: `-0.042048803633778095`
- Reconciled selected singleton gain: `1.0019046117862065`
- Reconciled off-context singleton gain: `-0.13995217362127335`

## Rationale

The active rank-matched contextual top-k-1 bracket remains useful as a low-churn functional-retention bracket. The reconciled singleton evidence replaces the stale global negative-singleton blocker with a narrower interpretation: in-context router-selected singletons are beneficial on average, but forced off-context singleton reuse is harmful. That supports context-gated singleton efficacy with off-context interference, not a broad reusable singleton causal-retention claim.

## Evidence Separation

- Support identity churn: exact support-set churn from the completed probe packets.
- Functional/logit churn: anchor logit MSE drift after transfer.
- Causal gain/regret: when present, the selected/oracle/off-context singleton reconciliation supersedes the older source singleton-gain caveat.
- CE guardrail: positive anchor CE deterioration must stay within `0.05`.
- Finite-update commutator: A-to-B versus B-to-A final-function logit MSE when present.

## Next Step

use the reconciled functional-retention bracket to choose a bounded backend repeat only if backend-stable retention evidence is needed for the next causal-retention claim
