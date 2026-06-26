# Active Top-k-1 Next-Evidence Selection

- Status: `pass`
- Decision: `active_topk1_next_evidence_selected`
- Selected experiment: `retention_churn`
- Requires GPU now: `False`
- New training required: `False`
- Matched-control coverage adequate: `True`
- Top-k-2 causal cooperation blocked: `True`
- Git commit: `0bc6962e0315b0453ebdeb793f95b2b8ca04a3a7`

## Rationale

The matched-control coverage is adequate and the active rank-matched top-k-1 bracket already has favorable support churn, functional/logit churn, finite-update commutator, transfer, and dense-control evidence. The finite-update-augmented causal gate keeps top-k-2 causal-cooperation claims blocked, so the highest-information follow-up is retention and functional churn rather than another mitigation family or CE variant.

## Next Step

run the retention/functional-churn follow-up as the next bounded branch: promoted contextual top-k-2, rank-matched contextual top-k-1, dense active-rank, and random fixed top-k-2 controls; prioritize anchor drift, functional churn, support identity churn, finite-update commutator risk, residual/logit deltas, and CE guardrails

## Strategy Review

- Present: `True`
- Strategic change level: `minor`
- Notify Ben: `False`
- Ben notification required: `False`
- Incorporation: accepted: the report follows the recommendation to run a no-training active-rank-matched top-k-1 selection report and favors retention/functional-churn when matched-control coverage is adequate
