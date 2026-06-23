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
| char_larger_hep_temporal_clipped_objective_gate_seed3 | supervised_ce | False | True | ok | 3.77074385 | 3.50802588 | -0.26271797 | 0.93032728 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3 | supervised_ce | False | True | ok | 3.77074385 | 3.10037756 | -0.67036629 | 0.82221909 | 0.01000000 | 0.66210938 | 0.00134552 |
| token_larger_hep_temporal_clipped_objective_gate_seed3 | supervised_ce | False | True | ok | 4.27324104 | 4.17101955 | -0.10222149 | 0.97607870 | 0.01000000 | 0.00000000 | 0.00000000 |
| token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3 | supervised_ce | False | True | ok | 4.27324104 | 3.59574461 | -0.67749643 | 0.84145607 | 0.01000000 | 0.55078125 | 0.01390111 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate_seed3`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed3/runs/char_larger_hep_temporal_clipped_objective_gate_seed3`
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed3/runs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`
- `token_larger_hep_temporal_clipped_objective_gate_seed3`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed3/runs/token_larger_hep_temporal_clipped_objective_gate_seed3`
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`: `results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed3/runs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate_seed3`: alpha 0.0: loss 3.50802588, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.50801826, delta 0.00017768, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.50801110, delta 0.00035551, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.50799632, delta 0.00071070, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`: alpha 0.0: loss 3.10037756, delta 0.00000000, support-change 0.63281250, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.10038066, delta 0.00018287, support-change 0.63281250, pinned-vs-repicked 0.00033510, alpha 0.5: loss 3.10038376, delta 0.00036478, support-change 0.63281250, pinned-vs-repicked 0.00067133, alpha 1.0: loss 3.10039020, delta 0.00073028, support-change 0.66210938, pinned-vs-repicked 0.00134552
- `token_larger_hep_temporal_clipped_objective_gate_seed3`: alpha 0.0: loss 4.17101955, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.17100859, delta 0.00022030, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.17099762, delta 0.00043964, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.17097569, delta 0.00088000, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3`: alpha 0.0: loss 3.59574461, delta 0.00000000, support-change 0.55078125, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.59575772, delta 0.00023222, support-change 0.55078125, pinned-vs-repicked 0.00347695, alpha 0.5: loss 3.59576988, delta 0.00046539, support-change 0.55078125, pinned-vs-repicked 0.00695282, alpha 1.0: loss 3.59579611, delta 0.00093031, support-change 0.55078125, pinned-vs-repicked 0.01390111

## Verdict

- Best HEP alpha by loss: `0.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3` with loss `3.10037756` and ordinary-logit delta `0.00000000`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_hep_temporal_clipped_objective_gate_seed3` with loss improvement `0.00002956` and ordinary-logit delta `0.00071070`
