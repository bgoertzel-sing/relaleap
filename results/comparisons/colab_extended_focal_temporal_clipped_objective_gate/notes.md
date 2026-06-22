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
| char_extended_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72165346 | 3.56889176 | -0.15276170 | 0.95895327 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_extended_focal_hep_temporal_clipped_objective_gate | supervised_ce_focal | False | True | ok | 3.53409672 | 3.35723019 | -0.17686653 | 0.94995425 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_extended_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_extended_focal_temporal_clipped_objective_gate/runs/char_extended_hep_temporal_clipped_objective_gate`
- `char_extended_focal_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_extended_focal_temporal_clipped_objective_gate/runs/char_extended_focal_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_extended_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.56889176, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.56887603, delta 0.00011921, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.56886053, delta 0.00023818, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.56882930, delta 0.00047636, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_extended_focal_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.56848574, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.56847024, delta 0.00012004, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.56845474, delta 0.00023985, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.56842327, delta 0.00047898, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_extended_focal_hep_temporal_clipped_objective_gate` with loss `3.56842327` and ordinary-logit delta `0.00047898`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_extended_focal_hep_temporal_clipped_objective_gate` with loss improvement `0.00006247` and ordinary-logit delta `0.00047898`
