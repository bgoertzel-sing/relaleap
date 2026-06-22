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
| char_xlarge_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.65675950 | 3.31777120 | -0.33898830 | 0.90729817 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_xlarge_focal_hep_temporal_clipped_objective_gate | supervised_ce_focal | False | True | ok | 3.46068215 | 3.05805540 | -0.40262675 | 0.88365682 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_xlarge_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/runs/char_xlarge_hep_temporal_clipped_objective_gate`
- `char_xlarge_focal_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_char_xlarge_focal_temporal_clipped_objective_gate/runs/char_xlarge_focal_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_xlarge_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.31777120, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.31776786, delta 0.00010979, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.31776357, delta 0.00021932, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.31775713, delta 0.00043821, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_xlarge_focal_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.31715393, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.31715107, delta 0.00011310, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.31714702, delta 0.00022632, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.31713986, delta 0.00045246, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_xlarge_focal_hep_temporal_clipped_objective_gate` with loss `3.31713986` and ordinary-logit delta `0.00045246`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_xlarge_focal_hep_temporal_clipped_objective_gate` with loss improvement `0.00001407` and ordinary-logit delta `0.00045246`
