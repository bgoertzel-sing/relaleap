# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `8` checked, passed `True`
- Artifact invariants: `6` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| token_larger_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 4.23026133 | 4.14451742 | -0.08574391 | 0.97973082 | 0.01000000 | 0.00000000 | 0.00000000 |
| token_larger_focal_hep_temporal_clipped_objective_gate_seed2 | supervised_ce_focal | False | True | ok | 4.09823704 | 3.99827075 | -0.09996629 | 0.97560749 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `token_larger_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2/runs/token_larger_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_focal_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_token_larger_focal_temporal_clipped_objective_gate_seed2/runs/token_larger_focal_hep_temporal_clipped_objective_gate_seed2`

## HEP Alpha Sweeps

- `token_larger_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 4.14451742, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.14450169, delta 0.00022990, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.14448500, delta 0.00045943, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.14445305, delta 0.00091887, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `token_larger_focal_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 4.14528656, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.14526939, delta 0.00024593, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.14525366, delta 0.00049186, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.14522123, delta 0.00098312, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `token_larger_hep_temporal_clipped_objective_gate_seed2` with loss `4.14445305` and ordinary-logit delta `0.00091887`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `token_larger_hep_temporal_clipped_objective_gate_seed2` with loss improvement `0.00006437` and ordinary-logit delta `0.00091887`
