# Support Width Deconfounding Audit

- Status: `pass`
- Decision: `run_colab_support_width_deconfounding_audit`
- Selected next direction: `colab_support_width_deconfounding_audit`
- Promote support-width default: `False`
- Comparison: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate`
- Artifact check: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/artifact_check_local.json`

## Rationale

The local validation-scale deconfounding matrix is artifact-backed. Top-k 2 improves best temporal-clipped HEP loss and support utilization relative to the top-k 1 baseline, while doubling columns at top-k 1 leaves the support audit collapsed onto one column.

## Matrix

| Variant | Columns | Top-k | Best HEP loss | Final residual loss | Used columns | Dead columns | Unique supports | Max column fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 12 | 1 | 3.58667445 | 3.58673668 | 1 | 11 | 1 | 1.00000000 |
| support_width | 12 | 2 | 3.49384785 | 3.49390101 | 10 | 2 | 15 | 0.37500000 |
| capacity | 24 | 1 | 3.58667445 | 3.58673668 | 1 | 23 | 1 | 1.00000000 |
| capacity_support_width | 24 | 2 | 3.49411678 | 3.49417019 | 15 | 9 | 17 | 0.35937500 |

## Comparisons

- Support width minus baseline best HEP loss: `-0.09282660`
- Capacity minus baseline best HEP loss: `0.00000000`
- Capacity+support width minus baseline best HEP loss: `-0.09255767`
- Support width minus baseline used columns: `9.0`
- Capacity minus baseline used columns: `0.0`

## Next Step

run the matching Colab support-width deconfounding audit through the real-Chrome CDP bridge
