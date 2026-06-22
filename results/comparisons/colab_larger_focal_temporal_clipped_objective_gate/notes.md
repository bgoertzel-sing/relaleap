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
| char_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.39831114 | -0.31763148 | 0.91452196 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_larger_focal_hep_temporal_clipped_objective_gate | supervised_ce_focal | False | True | ok | 3.52391744 | 3.15012836 | -0.37378908 | 0.89392797 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_larger_focal_temporal_clipped_objective_gate/runs/char_larger_hep_temporal_clipped_objective_gate`
- `char_larger_focal_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_larger_focal_temporal_clipped_objective_gate/runs/char_larger_focal_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.39831114, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.39830112, delta 0.00009918, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.39829135, delta 0.00019979, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.39827180, delta 0.00039887, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_larger_focal_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.39673686, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.39672732, delta 0.00010157, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.39671755, delta 0.00020337, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.39669800, delta 0.00040531, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_focal_hep_temporal_clipped_objective_gate` with loss `3.39669800` and ordinary-logit delta `0.00040531`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_focal_hep_temporal_clipped_objective_gate` with loss improvement `0.00003886` and ordinary-logit delta `0.00040531`
