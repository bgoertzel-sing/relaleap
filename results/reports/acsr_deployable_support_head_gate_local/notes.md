# ACSR Deployable Support-Head Gate

- Status: `pass`
- Decision: `deployable_support_head_gate_blocks_claim_pending_nulls_or_headroom`
- Claim status: `deployable_support_discovery_not_established_sparse_identity_retired`
- Learned-head heldout loss delta vs router: `-0.00047850608825683594`
- Learned-head heldout oracle-gap recovery: `0.1893932244974993`
- Upstream oracle CE headroom: `-0.0023670196533203125`

No major strategy-review direction shift recorded for this gate.

This is a local command-driven gate. It does not run RunPod, and it does not revive the sparse-support identity claim.

## Claim Blockers
- `learned_head_recovers_oracle_gap`: learned support head recovery is too small
- `oracle_support_headroom_positive`: upstream oracle CE headroom is too small for a deployable target
