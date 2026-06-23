# Exhaustive Support Audit Decision

- Status: `pass`
- Decision: `diagnose_exhaustive_support_audit`
- Selected next direction: `router_support_selection`
- Audit: `results/audits/validation_support_wide_exhaustive_support`
- Summary: `results/audits/validation_support_wide_exhaustive_support/summary.json`

## Rationale

The per-token oracle beats the learned router on most evaluated positions, so the strongest signal is router support-selection headroom: regret=0.03597923, positive_fraction=0.87301588. The learned router still beats the best single global fixed pair by 0.03413963, which points to improving token-conditioned routing rather than replacing it with one fixed support.

## Evidence

- Router loss: `3.49828529`
- Per-token oracle loss: `3.46230602`
- Oracle-support regret: `0.03597923`
- Oracle-support positive fraction: `0.87301588`
- Router minus best global fixed support loss: `-0.03413963`
- Best global fixed support: `1,2`
- Dominant router support: `0,1`
- Best one-swap support: `1,2`
- Router-target holdout accuracy: `0.40476191`
- Router-target holdout selector minus router loss: `0.04427838`
- Router-target holdout oracle-gap recovery: `-1.06762746`
- Router-target nonlinear holdout accuracy: `0.39682540`
- Router-target nonlinear holdout selector minus router loss: `0.03769732`
- Router-target nonlinear holdout oracle-gap recovery: `-0.90894667`
- Router-target contextual holdout accuracy: `0.89682537`
- Router-target contextual holdout selector minus router loss: `-0.03149462`
- Router-target contextual holdout oracle-gap recovery: `0.75938903`
- Best router-target holdout oracle-gap recovery: `0.75938903`
- Used columns: `8` of `12`
- Dead columns: `4`
- Unique router support sets: `11`
- Strongest pairwise synergy: `0.06710577`

## Signals

- Router improvement: `True`
- Column redundancy: `True`
- Pairwise composition: `True`

## Next Step

repeat the best router oracle-target selector, including contextual features, on a fresh seed and larger support-width setting
