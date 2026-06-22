# Temporal Clipped HEP Decision Report

- Status: `pass`
- Decision: `select_temporal_label_free_support_stress_candidate`
- Selected label-free support-stress candidate: `True`
- Promote to default support-stress mitigation: `False`
- Deployable label-free signal: `True`
- Artifact check: `pass`
- Comparison verdict: `pass`
- Max support change fraction: `0.60546875`
- Max temporal pinned-vs-repicked logit delta: `0.00397205`

## Rationale

Temporal clipped HEP is deployable at inference time and produced a nonzero alpha with support-stress loss improvement inside both stability budgets, while entropy did not improve loss in the same comparison and the guided oracle remains diagnostic-only.

## Temporal Clipped HEP Candidates

| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000000 | 4.53275919 | 0.00000000 | 0.00000000 | 0.60546875 | 0.00000000 |
| 0.25000000 | 4.53273249 | 0.00002670 | 0.00043538 | 0.60546875 | 0.00099468 |
| 0.50000000 | 4.53270388 | 0.00005531 | 0.00087079 | 0.60546875 | 0.00198698 |
| 1.00000000 | 4.53264904 | 0.00011015 | 0.00174144 | 0.60546875 | 0.00397205 |

## Next Step

use temporal consistency as the selected label-free candidate for the next support-stress mitigation experiment
