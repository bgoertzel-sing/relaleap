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
| char_xxlarge_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.76073313 | 3.38818288 | -0.37255025 | 0.90093680 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2 | supervised_ce_focal | False | True | ok | 3.57782531 | 3.13634062 | -0.44148469 | 0.87660530 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_xxlarge_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2/runs/char_xxlarge_hep_temporal_clipped_objective_gate_seed2`
- `char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate_seed2/runs/char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2`

## HEP Alpha Sweeps

- `char_xxlarge_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.38818288, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.38817716, delta 0.00010467, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.38817167, delta 0.00020897, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.38816094, delta 0.00041747, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.38746929, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.38746405, delta 0.00010777, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.38745809, delta 0.00021613, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.38744736, delta 0.00043213, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2` with loss `3.38744736` and ordinary-logit delta `0.00043213`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2` with loss improvement `0.00002193` and ordinary-logit delta `0.00043213`
