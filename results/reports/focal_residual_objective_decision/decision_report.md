# Focal Residual Objective Decision

- Status: `fail`
- Decision: `insufficient_evidence`
- Continue focal validation: `False`
- Selected variant: `None`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean focal minus supervised best HEP loss: `-0.00065478`
- Mean focal minus supervised final residual loss: `-0.21224785`

## Rationale

The focal decision requires matching local and Colab comparisons with passing artifact checks and valid supervised and focal temporal-clipped runs.

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
| colab |  |  |  |  |  | `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate` |

## Failures

- `comparison.summary.json` expected `file exists`, got `missing` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/summary.json`
- `artifact_check.status` expected `pass`, got `None` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/artifact_check_local.json`
- `comparison.status` expected `ok`, got `None` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate`
- `comparison.verdict.status` expected `pass`, got `None` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate`
- `comparison.runs.supervised_ce` expected `one run`, got `0` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate`
- `comparison.runs.supervised_ce_focal` expected `one run`, got `0` at `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate`

## Next Step

run or extract the missing matching focal objective comparison artifacts
