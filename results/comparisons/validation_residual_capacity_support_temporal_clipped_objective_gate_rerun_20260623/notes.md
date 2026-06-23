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
| char_validation_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.58673668 | -0.14213300 | 0.96188309 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_capacity_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.58673668 | -0.14213300 | 0.96188309 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.48482919 | -0.24404049 | 0.93455376 | 0.01000000 | 0.41015625 | 0.00322439 |
| char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.48526955 | -0.24360013 | 0.93467186 | 0.01000000 | 0.28906250 | 0.00147097 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_rerun_20260623/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_rerun_20260623/runs/char_validation_capacity_hep_temporal_clipped_objective_gate`
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_rerun_20260623/runs/char_validation_support_wide_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_rerun_20260623/runs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012290, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024533, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012290, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024533, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.48482919, delta 0.00000000, support-change 0.41015625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.48481822, delta 0.00011790, support-change 0.41015625, pinned-vs-repicked 0.00080538, alpha 0.5: loss 3.48480654, delta 0.00023675, support-change 0.41015625, pinned-vs-repicked 0.00161117, alpha 1.0: loss 3.48478436, delta 0.00047338, support-change 0.41015625, pinned-vs-repicked 0.00322439
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.48526955, delta 0.00000000, support-change 0.28515625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.48525763, delta 0.00012112, support-change 0.28515625, pinned-vs-repicked 0.00036737, alpha 0.5: loss 3.48524547, delta 0.00024248, support-change 0.28515625, pinned-vs-repicked 0.00073470, alpha 1.0: loss 3.48522139, delta 0.00048496, support-change 0.28906250, pinned-vs-repicked 0.00147097

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_support_wide_hep_temporal_clipped_objective_gate` with loss `3.48478436` and ordinary-logit delta `0.00047338`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_support_wide_hep_temporal_clipped_objective_gate` with loss improvement `0.00004482` and ordinary-logit delta `0.00047338`
