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
| char_validation_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.48881698 | -0.24005270 | 0.93562320 | 0.01000000 | 0.35546875 | 0.00292814 |
| char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.47731900 | -0.25155068 | 0.93253970 | 0.01000000 | 0.37500000 | 0.00340658 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_tiebreak_20260623/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_tiebreak_20260623/runs/char_validation_capacity_hep_temporal_clipped_objective_gate`
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_tiebreak_20260623/runs/char_validation_support_wide_hep_temporal_clipped_objective_gate`
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_residual_capacity_support_temporal_clipped_objective_gate_tiebreak_20260623/runs/char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012290, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024533, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_capacity_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012290, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024533, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.48881698, delta 0.00000000, support-change 0.35156250, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.48880577, delta 0.00011420, support-change 0.35156250, pinned-vs-repicked 0.00073159, alpha 0.5: loss 3.48879433, delta 0.00022840, support-change 0.35156250, pinned-vs-repicked 0.00146306, alpha 1.0: loss 3.48877120, delta 0.00045633, support-change 0.35546875, pinned-vs-repicked 0.00292814
- `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.47731900, delta 0.00000000, support-change 0.37500000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.47730780, delta 0.00011992, support-change 0.37500000, pinned-vs-repicked 0.00085142, alpha 0.5: loss 3.47729540, delta 0.00024045, support-change 0.37500000, pinned-vs-repicked 0.00170317, alpha 1.0: loss 3.47727180, delta 0.00048101, support-change 0.37500000, pinned-vs-repicked 0.00340658

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate` with loss `3.47727180` and ordinary-logit delta `0.00048101`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate` with loss improvement `0.00004721` and ordinary-logit delta `0.00048101`
