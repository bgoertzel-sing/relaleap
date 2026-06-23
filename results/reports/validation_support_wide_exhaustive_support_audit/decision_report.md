# Exhaustive Support Audit Decision

- Status: `pass`
- Decision: `diagnose_exhaustive_support_audit`
- Selected next direction: `router_support_selection`
- Audit: `results/audits/validation_support_wide_exhaustive_support`
- Summary: `results/audits/validation_support_wide_exhaustive_support/summary.json`

## Rationale

The per-token oracle beats the learned router on most evaluated positions, so the strongest signal is router support-selection headroom: regret=0.04464032, positive_fraction=0.87698412. The learned router still beats the best single global fixed pair by 0.07553172, which points to improving token-conditioned routing rather than replacing it with one fixed support.

## Evidence

- Router loss: `3.48559761`
- Per-token oracle loss: `3.44095731`
- Oracle-support regret: `0.04464032`
- Oracle-support positive fraction: `0.87698412`
- Router minus best global fixed support loss: `-0.07553172`
- Best global fixed support: `1,3`
- Dominant router support: `0,1`
- Best one-swap support: `1,3`
- Router-target holdout accuracy: `0.38095239`
- Router-target holdout selector minus router loss: `0.04837275`
- Router-target holdout oracle-gap recovery: `-1.03543305`
- Used columns: `10` of `12`
- Dead columns: `2`
- Unique router support sets: `16`
- Strongest pairwise synergy: `0.04164767`

## Signals

- Router improvement: `True`
- Column redundancy: `True`
- Pairwise composition: `True`

## Next Step

prototype a contextual or nonlinear router-support diagnostic because the linear oracle-target selector did not recover the holdout oracle gap
