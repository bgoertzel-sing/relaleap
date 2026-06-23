# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `20` checked, passed `True`
- Artifact invariants: `15` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_validation_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886968 | 3.58673668 | -0.14213300 | 0.96188309 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_temporal_consistency_hep_temporal_clipped_objective_gate | supervised_ce_temporal_consistency | False | True | ok | 3.76521182 | 3.62303972 | -0.14217210 | 0.96224061 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_temporal_consistency_w005_hep_temporal_clipped_objective_gate | supervised_ce_temporal_consistency | False | True | ok | 3.91058016 | 3.76825190 | -0.14232826 | 0.96360431 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_temporal_consistency_w010_hep_temporal_clipped_objective_gate | supervised_ce_temporal_consistency | False | True | ok | 4.09229040 | 3.94976640 | -0.14252400 | 0.96517256 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate | supervised_ce_temporal_consistency | False | True | ok | 4.45571136 | 4.31279516 | -0.14291620 | 0.96792517 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_temporal_consistency_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/runs/char_validation_temporal_consistency_hep_temporal_clipped_objective_gate`
- `char_validation_temporal_consistency_w005_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/runs/char_validation_temporal_consistency_w005_hep_temporal_clipped_objective_gate`
- `char_validation_temporal_consistency_w010_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/runs/char_validation_temporal_consistency_w010_hep_temporal_clipped_objective_gate`
- `char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate`: `results/comparisons/validation_temporal_consistency_weight_sweep_temporal_clipped_objective_gate/runs/char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012290, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024533, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_temporal_consistency_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673620, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672094, delta 0.00012302, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024617, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049150, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_temporal_consistency_w005_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673429, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58671880, delta 0.00012314, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670306, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667231, delta 0.00049114, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_temporal_consistency_w010_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673167, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58671618, delta 0.00012326, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670068, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58666992, delta 0.00049174, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58672690, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58671165, delta 0.00012302, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58669591, delta 0.00024545, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58666468, delta 0.00049126, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate` with loss `3.58666468` and ordinary-logit delta `0.00049126`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate` with loss improvement `0.00006223` and ordinary-logit delta `0.00049126`
