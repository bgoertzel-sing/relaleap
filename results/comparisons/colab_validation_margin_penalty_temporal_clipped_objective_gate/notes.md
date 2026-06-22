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
| char_validation_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.58673668 | -0.14213252 | 0.96188321 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_margin_penalty_hep_temporal_clipped_objective_gate | supervised_ce_margin_penalty | False | True | ok | 3.74445653 | 3.60099769 | -0.14345884 | 0.96168767 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_margin_penalty_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate/runs/char_validation_margin_penalty_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012314, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049138, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_margin_penalty_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58695531, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58693981, delta 0.00012112, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58692408, delta 0.00024307, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58689332, delta 0.00048578, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_hep_temporal_clipped_objective_gate` with loss `3.58667445` and ordinary-logit delta `0.00049138`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_hep_temporal_clipped_objective_gate` with loss improvement `0.00006223` and ordinary-logit delta `0.00049138`
