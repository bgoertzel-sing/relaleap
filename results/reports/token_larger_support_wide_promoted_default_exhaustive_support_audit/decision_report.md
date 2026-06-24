# Exhaustive Support Audit Decision

- Status: `pass`
- Decision: `diagnose_exhaustive_support_audit`
- Selected next direction: `column_redundancy`
- Audit: `results/audits/token_larger_support_wide_promoted_default_exhaustive_support`
- Summary: `results/audits/token_larger_support_wide_promoted_default_exhaustive_support/summary.json`

## Rationale

The audit does not show dominant oracle-router regret, but support utilization is incomplete: used_columns=20, num_columns=24, dead_columns=4. The next branch should separate redundancy from routing load balance.

## Evidence

- Router loss: `2.86810708`
- Per-token oracle loss: `2.86557889`
- Oracle-support regret: `0.00252840`
- Oracle-support positive fraction: `0.05555556`
- Router minus best global fixed support loss: `-1.13105989`
- Best global fixed support: `0,3`
- Dominant router support: `14,18`
- Best one-swap support: `0,14`
- Router-target holdout accuracy: `0.81746030`
- Router-target holdout selector minus router loss: `0.17577481`
- Router-target holdout oracle-gap recovery: `-69.57185996`
- Router-target nonlinear holdout accuracy: `0.80952382`
- Router-target nonlinear holdout selector minus router loss: `0.18509197`
- Router-target nonlinear holdout oracle-gap recovery: `-73.25960177`
- Router-target contextual holdout accuracy: `0.96825397`
- Router-target contextual holdout selector minus router loss: `-0.00252652`
- Router-target contextual holdout oracle-gap recovery: `1.00000000`
- Contextual support-intervention holdout loss: `2.87246418`
- Contextual support-intervention holdout minus router loss: `-0.00252652`
- Contextual support-intervention holdout oracle-gap recovery: `1.00000000`
- Contextual support-head holdout loss: `2.87451220`
- Contextual support-head holdout minus router loss: `-0.00047851`
- Contextual support-head holdout oracle-gap recovery: `0.18939322`
- Best router-target holdout oracle-gap recovery: `1.00000000`
- Used columns: `20` of `24`
- Dead columns: `4`
- Unique router support sets: `50`
- Strongest pairwise synergy: `0.25717735`

## Signals

- Router improvement: `False`
- Column redundancy: `True`
- Pairwise composition: `True`

## Next Step

prototype a column redundancy/load-balancing diagnostic on the same support-width setting
