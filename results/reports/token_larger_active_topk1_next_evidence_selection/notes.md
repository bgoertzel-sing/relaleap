# Active Top-k-1 Next-Evidence Selection

- Status: `pass`
- Decision: `active_topk1_next_evidence_selected`
- Selected experiment: `context_gate_suppression_calibration_audit`
- Claim status: `column_plus_context_gate_hypothesis`
- Claim policy: `broad_reusable_singleton_claim_excluded`
- Requires GPU now: `False`
- New training required: `False`
- Git commit: `1107825193739fe41d75cb854bdf42338128e899`

## Key Evidence

- Own-context singleton gain mean: `1.0019046117862065`
- Off-context singleton gain mean: `-0.13995217362127335`
- Context-gated holdout net gain: `0.7776962748832172`
- Context gate minus ungated holdout: `0.44408054031165584`

## Rationale

The RunPod-validated decomposition already establishes positive own-context singleton gain, negative off-context forced-singleton reuse, and positive context-gated holdout gain. Repeating backend validation or adding another wrapper would duplicate completed work. The next useful evidence is a bounded no-training gate-suppression audit that tests whether off-context interference can be cleanly gated away while retaining the in-context singleton gain.

## Next Step

implement and run the local no-training context-gate suppression calibration audit using the validated interference CSV artifacts

## Strategy Review

- Present: `True`
- Strategic change level: `minor`
- Notify Ben: `False`
- Ben notification required: `False`
- Incorporation: followed where still applicable: the recommended context-conditioned interference decomposition and RunPod closeout are complete; this report selects the next bounded local gate-suppression audit rather than duplicating backend validation
