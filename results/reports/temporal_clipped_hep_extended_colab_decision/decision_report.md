# Temporal Clipped HEP Decision Report

- Status: `pass`
- Decision: `select_temporal_label_free_support_stress_candidate`
- Selected label-free support-stress candidate: `True`
- Promote to default support-stress mitigation: `False`
- Deployable label-free signal: `True`
- Artifact check: `pass`
- Comparison verdict: `pass`
- Max support change fraction: `0.63802083`
- Max temporal pinned-vs-repicked logit delta: `0.00556183`

## Rationale

Temporal clipped HEP is deployable at inference time and produced a nonzero alpha with support-stress loss improvement inside both stability budgets, while entropy did not improve loss in the same comparison and the guided oracle remains diagnostic-only.

## Temporal Clipped HEP Candidates

| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000000 | 4.63250589 | 0.00000000 | 0.00000000 | 0.63802083 | 0.00000000 |
| 0.25000000 | 4.63244867 | 0.00005722 | 0.00059482 | 0.63802083 | 0.00139189 |
| 0.50000000 | 4.63239145 | 0.00011444 | 0.00118989 | 0.63802083 | 0.00278187 |
| 1.00000000 | 4.63227940 | 0.00022650 | 0.00237951 | 0.63802083 | 0.00556183 |

## Next Step

use temporal consistency as the selected label-free candidate for the next support-stress mitigation experiment
