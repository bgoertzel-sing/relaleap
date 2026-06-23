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
| char_larger_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864539 | 3.36594748 | -0.29269791 | 0.91999828 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864539 | 3.14497447 | -0.51367092 | 0.85960079 | 0.01000000 | 0.44726562 | 0.00270697 |
| token_larger_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 4.23026133 | 4.14451742 | -0.08574391 | 0.97973082 | 0.01000000 | 0.00000000 | 0.00000000 |
| token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 4.23026133 | 3.59730530 | -0.63295603 | 0.85037425 | 0.01000000 | 0.39843750 | 0.00172412 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/char_larger_hep_temporal_clipped_objective_gate_seed2`
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/token_larger_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.36594748, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.36594224, delta 0.00008422, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.36593652, delta 0.00016826, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.36592674, delta 0.00033671, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.14497447, delta 0.00000000, support-change 0.44531250, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.14497447, delta 0.00011584, support-change 0.44531250, pinned-vs-repicked 0.00067425, alpha 0.5: loss 3.14497399, delta 0.00023204, support-change 0.44531250, pinned-vs-repicked 0.00135016, alpha 1.0: loss 3.14497328, delta 0.00046399, support-change 0.44726562, pinned-vs-repicked 0.00270697
- `token_larger_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 4.14451742, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.14450169, delta 0.00023013, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.14448500, delta 0.00045955, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.14445353, delta 0.00091887, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.59730530, delta 0.00000000, support-change 0.39843750, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.59731269, delta 0.00022650, support-change 0.39843750, pinned-vs-repicked 0.00042892, alpha 0.5: loss 3.59732008, delta 0.00045276, support-change 0.39843750, pinned-vs-repicked 0.00085938, alpha 1.0: loss 3.59733510, delta 0.00090599, support-change 0.39843750, pinned-vs-repicked 0.00172412

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2` with loss `3.14497328` and ordinary-logit delta `0.00046399`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2` with loss improvement `0.00000119` and ordinary-logit delta `0.00046399`
