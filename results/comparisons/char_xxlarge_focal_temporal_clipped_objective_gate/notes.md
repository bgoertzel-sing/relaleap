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
| char_xxlarge_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.70722270 | 3.35724783 | -0.34997487 | 0.90559648 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_xxlarge_focal_hep_temporal_clipped_objective_gate | supervised_ce_focal | False | True | ok | 3.51527071 | 3.10445833 | -0.41081238 | 0.88313492 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_xxlarge_hep_temporal_clipped_objective_gate`: `results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate/runs/char_xxlarge_hep_temporal_clipped_objective_gate`
- `char_xxlarge_focal_hep_temporal_clipped_objective_gate`: `results/comparisons/char_xxlarge_focal_temporal_clipped_objective_gate/runs/char_xxlarge_focal_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_xxlarge_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.35724783, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.35724044, delta 0.00011230, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.35723233, delta 0.00022459, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.35721779, delta 0.00044942, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_xxlarge_focal_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.35650802, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.35650063, delta 0.00011349, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.35649252, delta 0.00022817, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.35647798, delta 0.00045729, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_xxlarge_focal_hep_temporal_clipped_objective_gate` with loss `3.35647798` and ordinary-logit delta `0.00045729`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_xxlarge_focal_hep_temporal_clipped_objective_gate` with loss improvement `0.00003004` and ordinary-logit delta `0.00045729`
