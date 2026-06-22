# Focal Residual Objective Decision

- Status: `pass`
- Decision: `continue_focal_residual_objective_validation`
- Continue focal validation: `True`
- Selected variant: `supervised_ce_focal`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean focal minus supervised best HEP loss: `-0.00019383`
- Mean focal minus supervised final residual loss: `-0.20871329`

## Rationale

The focal objective beats supervised CE HEP loss in at least one artifact-backed backend, so it merits broader objective validation before any default change.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | Focal best HEP loss | Focal minus supervised | Focal final residual loss | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.58648062 | -0.00019383 | 3.37802339 | `results/comparisons/validation_focal_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.58648062 | -0.00019383 | 3.37802339 | `results/comparisons/colab_validation_focal_temporal_clipped_objective_gate` |

## Next Step

run a broader focal objective comparison outside the current char validation setting
