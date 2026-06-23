# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `16` checked, passed `True`
- Artifact invariants: `12` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_validation_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.58673668 | -0.14213252 | 0.96188321 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_capacity_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.58673668 | -0.14213252 | 0.96188321 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.49390101 | -0.23496819 | 0.93698674 | 0.01000000 | 0.30859375 | 0.00819236 |
| char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.49417019 | -0.23469901 | 0.93705893 | 0.01000000 | 0.40625000 | 0.00652322 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/runs/char_validation_capacity_hep_temporal_clipped_objective_gate`
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/runs/char_validation_support_wide_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_residual_capacity_support_temporal_clipped_objective_gate/runs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012314, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049138, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012314, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049138, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.49390101, delta 0.00000000, support-change 0.30859375, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.49388790, delta 0.00012052, support-change 0.30859375, pinned-vs-repicked 0.00204748, alpha 0.5: loss 3.49387479, delta 0.00024164, support-change 0.30859375, pinned-vs-repicked 0.00409544, alpha 1.0: loss 3.49384785, delta 0.00048268, support-change 0.30859375, pinned-vs-repicked 0.00819236
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.49417019, delta 0.00000000, support-change 0.40625000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.49415684, delta 0.00012124, support-change 0.40625000, pinned-vs-repicked 0.00163037, alpha 0.5: loss 3.49414349, delta 0.00024259, support-change 0.40625000, pinned-vs-repicked 0.00326109, alpha 1.0: loss 3.49411678, delta 0.00048482, support-change 0.40625000, pinned-vs-repicked 0.00652322

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_support_wide_hep_temporal_clipped_objective_gate` with loss `3.49384785` and ordinary-logit delta `0.00048268`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_support_wide_hep_temporal_clipped_objective_gate` with loss improvement `0.00005317` and ordinary-logit delta `0.00048268`
