# Residual Objective Gate Decision

- Status: `pass`
- Decision: `keep_supervised_ce_residual_objective_default`
- Continue PC residual objective validation: `False`
- Promote residual learning method: `False`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Supervised run count: `2`
- PC run count: `2`
- PC CE win count: `0`
- Mean supervised best HEP loss: `3.58667445`
- Mean PC best HEP loss: `3.59590077`

## Rationale

The objective-discriminative local and Colab evidence is valid and both objectives improve their own training losses, but PC-style residual training does not beat supervised residual training on supervised CE HEP loss. The default residual objective should remain supervised CE.

## Evidence

| Backend | Artifact check | Verdict | Objective | Own loss delta | Best HEP alpha | Best HEP loss | Support preset disabled | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| local | pass | pass | supervised_ce | -0.14213300 | 1.00000000 | 3.58667445 | True | `results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate` |
| local | pass | pass | pc_logit_mse | -0.00029955 | 1.00000000 | 3.59590077 | True | `results/comparisons/validation_pc_vs_supervised_temporal_clipped_objective_gate` |
| colab | pass | pass | supervised_ce | -0.14213252 | 1.00000000 | 3.58667445 | True | `results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate` |
| colab | pass | pass | pc_logit_mse | -0.00029955 | 1.00000000 | 3.59590077 | True | `results/comparisons/colab_validation_pc_vs_supervised_temporal_clipped_objective_gate` |

## Next Step

inspect PC residual objective variants or diagnostics before another promotion-style objective gate
