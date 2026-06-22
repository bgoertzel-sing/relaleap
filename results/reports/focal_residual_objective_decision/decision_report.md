# Focal Residual Objective Decision

- Status: `pass`
- Decision: `continue_focal_residual_objective_validation`
- Continue focal validation: `True`
- Selected variant: `supervised_ce_focal`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean focal minus supervised best HEP loss: `-0.00065103`
- Mean focal minus supervised final residual loss: `-0.21699464`

## Rationale

The focal objective beats supervised CE HEP loss in every artifact-backed comparison, including the broader extended, larger, tokenized larger, and xlarge local and Colab checks, so it remains the selected objective variant for the next scale before any default change.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | Focal best HEP loss | Focal minus supervised | Focal final residual loss | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.58648062 | -0.00019383 | 3.37802339 | `results/comparisons/validation_focal_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.58648062 | -0.00019383 | 3.37802339 | `results/comparisons/colab_validation_focal_temporal_clipped_objective_gate` |
| local | pass | 3.56882930 | 3.56842327 | -0.00040603 | 3.35723019 | `results/comparisons/extended_focal_temporal_clipped_objective_gate` |
| colab | pass | 3.56882930 | 3.56842327 | -0.00040603 | 3.35723019 | `results/comparisons/colab_extended_focal_temporal_clipped_objective_gate` |
| local | pass | 3.39827180 | 3.39669800 | -0.00157380 | 3.15012836 | `results/comparisons/larger_focal_temporal_clipped_objective_gate` |
| colab | pass | 3.39827180 | 3.39669800 | -0.00157380 | 3.15012836 | `results/comparisons/colab_larger_focal_temporal_clipped_objective_gate` |
| local | pass | 4.05921125 | 4.05874681 | -0.00046444 | 3.90255094 | `results/comparisons/token_larger_focal_temporal_clipped_objective_gate` |
| colab | pass | 4.05921125 | 4.05874681 | -0.00046444 | 3.90255117 | `results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate` |
| local | pass | 3.31775665 | 3.31713986 | -0.00061679 | 3.05805564 | `results/comparisons/char_xlarge_focal_temporal_clipped_objective_gate` |
| colab | pass | 3.31775713 | 3.31713986 | -0.00061727 | 3.05805540 | `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate` |

## Next Step

run the next focal objective scale check under the same objective gate
