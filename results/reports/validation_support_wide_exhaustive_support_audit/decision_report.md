# Exhaustive Support Audit Decision

- Status: `pass`
- Decision: `diagnose_exhaustive_support_audit`
- Selected next direction: `router_support_selection`
- Audit: `results/audits/validation_support_wide_exhaustive_support`
- Summary: `results/audits/validation_support_wide_exhaustive_support/summary.json`

## Rationale

The per-token oracle beats the learned router on most evaluated positions, so the strongest signal is router support-selection headroom: regret=0.05123724, positive_fraction=0.95238096. The learned router still beats the best single global fixed pair by 0.06077313, which points to improving token-conditioned routing rather than replacing it with one fixed support.

## Evidence

- Router loss: `3.49678302`
- Per-token oracle loss: `3.44554567`
- Oracle-support regret: `0.05123724`
- Oracle-support positive fraction: `0.95238096`
- Router minus best global fixed support loss: `-0.06077313`
- Best global fixed support: `0,3`
- Dominant router support: `0,1`
- Best one-swap support: `0,3`
- Used columns: `9` of `12`
- Dead columns: `3`
- Unique router support sets: `15`
- Strongest pairwise synergy: `0.04036903`

## Signals

- Router improvement: `True`
- Column redundancy: `True`
- Pairwise composition: `True`

## Next Step

prototype a router-support improvement diagnostic that uses the exhaustive audit as the oracle target
