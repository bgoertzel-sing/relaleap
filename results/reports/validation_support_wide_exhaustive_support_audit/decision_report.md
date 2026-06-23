# Exhaustive Support Audit Decision

- Status: `pass`
- Decision: `diagnose_exhaustive_support_audit`
- Selected next direction: `router_support_selection`
- Audit: `results/audits/validation_support_wide_exhaustive_support`
- Summary: `results/audits/validation_support_wide_exhaustive_support/summary.json`

## Rationale

The per-token oracle beats the learned router on most evaluated positions, so the strongest signal is router support-selection headroom: regret=0.06139732, positive_fraction=0.93253970. The learned router still beats the best single global fixed pair by 0.03750205, which points to improving token-conditioned routing rather than replacing it with one fixed support.

## Evidence

- Router loss: `3.51750112`
- Per-token oracle loss: `3.45610356`
- Oracle-support regret: `0.06139732`
- Oracle-support positive fraction: `0.93253970`
- Router minus best global fixed support loss: `-0.03750205`
- Best global fixed support: `0,2`
- Dominant router support: `0,1`
- Best one-swap support: `0,2`
- Router-target holdout accuracy: `0.32539684`
- Router-target holdout selector minus router loss: `0.02856421`
- Router-target holdout oracle-gap recovery: `-0.43181163`
- Router-target nonlinear holdout accuracy: `0.33333334`
- Router-target nonlinear holdout selector minus router loss: `0.02387667`
- Router-target nonlinear holdout oracle-gap recovery: `-0.36094892`
- Best router-target holdout oracle-gap recovery: `-0.36094892`
- Used columns: `9` of `12`
- Dead columns: `3`
- Unique router support sets: `12`
- Strongest pairwise synergy: `0.01413918`

## Signals

- Router improvement: `True`
- Column redundancy: `True`
- Pairwise composition: `True`

## Next Step

prototype a contextual router-support diagnostic with token position or sequence-neighborhood features because hidden-only oracle-target selectors did not recover the holdout oracle gap
