# Temporal Clipped HEP Decision Report

- Status: `pass`
- Decision: `select_temporal_label_free_support_stress_candidate`
- Selected label-free support-stress candidate: `True`
- Promote to default support-stress mitigation: `False`
- Deployable label-free signal: `True`
- Artifact check: `pass`
- Comparison verdict: `pass`
- Max support change fraction: `0.65039062`
- Max temporal pinned-vs-repicked logit delta: `0.00319910`

## Rationale

Temporal clipped HEP is deployable at inference time and produced a nonzero alpha with support-stress loss improvement inside both stability budgets, while entropy did not improve loss in the same comparison and the guided oracle remains diagnostic-only.

## Temporal Clipped HEP Candidates

| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000000 | 3.92103243 | 0.00000000 | 0.00000000 | 0.65039062 | 0.00000000 |
| 0.25000000 | 3.92101884 | 0.00001359 | 0.00015724 | 0.65039062 | 0.00080085 |
| 0.50000000 | 3.92100549 | 0.00002694 | 0.00031438 | 0.65039062 | 0.00159979 |
| 1.00000000 | 3.92097831 | 0.00005412 | 0.00062868 | 0.65039062 | 0.00319910 |

## Next Step

use temporal consistency as the selected label-free candidate for the next support-stress mitigation experiment
