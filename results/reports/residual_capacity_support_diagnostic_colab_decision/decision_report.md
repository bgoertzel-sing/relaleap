# Residual Capacity Support Colab Decision

- Status: `fail`
- Decision: `insufficient_evidence`
- Selected direction: `None`
- Default residual objective: `supervised_ce`

## Rationale

The paired local/Colab diagnostic is missing, failing, or does not consistently select widened support under the policy.

## Evidence

| Backend | Artifact check | Verdict | Best variant | Support minus baseline | Accepted support alpha |
| --- | --- | --- | --- | ---: | ---: |
| local | `pass` | `pass` | `support_width` | -0.08517456 | 1.00000000 |
| colab | `pass` | `pass` | `baseline` | 0.00000072 | 1.00000000 |

## Failures

- `colab.support_width.best_hep_loss` expected `< baseline best HEP loss`, got `delta 7.152557373046875e-07` at `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate`
- `colab.best_variant` expected `support_width`, got `baseline` at `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate`

## Next Step

diagnose the local/Colab residual capacity/support divergence before any broader support-width gate
