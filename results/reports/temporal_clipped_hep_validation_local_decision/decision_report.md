# Temporal Clipped HEP Decision Report

- Status: `pass`
- Decision: `select_temporal_label_free_support_stress_candidate`
- Selected label-free support-stress candidate: `True`
- Promote to default support-stress mitigation: `False`
- Deployable label-free signal: `True`
- Artifact check: `pass`
- Comparison verdict: `pass`
- Max support change fraction: `0.59765625`
- Max temporal pinned-vs-repicked logit delta: `0.00378132`

## Rationale

Temporal clipped HEP is deployable at inference time and produced a nonzero alpha with support-stress loss improvement inside both stability budgets, while entropy did not improve loss in the same comparison and the guided oracle remains diagnostic-only.

## Temporal Clipped HEP Candidates

| Alpha | Loss | Improvement vs alpha 0 | Logit delta | Support change | Pinned-vs-repicked |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00000000 | 4.60237598 | 0.00000000 | 0.00000000 | 0.59765625 | 0.00000000 |
| 0.25000000 | 4.60231304 | 0.00006294 | 0.00060284 | 0.59765625 | 0.00094700 |
| 0.50000000 | 4.60224962 | 0.00012636 | 0.00120559 | 0.59765625 | 0.00189161 |
| 1.00000000 | 4.60212326 | 0.00025272 | 0.00241092 | 0.59765625 | 0.00378132 |

## Next Step

use temporal consistency as the selected label-free candidate for the next support-stress mitigation experiment
