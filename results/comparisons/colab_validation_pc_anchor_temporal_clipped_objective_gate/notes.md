# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `12` checked, passed `True`
- Artifact invariants: `9` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_validation_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.72886920 | 3.58673668 | -0.14213252 | 0.96188321 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_pc_hep_temporal_clipped_objective_gate | pc_logit_mse | False | True | ok | 0.02901663 | 0.02871708 | -0.00029955 | 0.98967661 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_validation_pc_anchor_hep_temporal_clipped_objective_gate | pc_logit_mse_ce_anchor | False | True | ok | 0.40190357 | 0.38741630 | -0.01448727 | 0.96395337 | 0.01000000 | 0.00000000 | 0.00000000 |

## Artifacts

- `char_validation_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_hep_temporal_clipped_objective_gate`
- `char_validation_pc_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_pc_hep_temporal_clipped_objective_gate`
- `char_validation_pc_anchor_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate/runs/char_validation_pc_anchor_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_validation_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58673668, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58672142, delta 0.00012314, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58670545, delta 0.00024569, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58667445, delta 0.00049138, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_pc_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.59595990, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.59594512, delta 0.00012124, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.59592986, delta 0.00024223, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.59590077, delta 0.00048423, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_validation_pc_anchor_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.58678913, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58677387, delta 0.00012231, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.58675814, delta 0.00024426, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.58672714, delta 0.00048864, support-change 0.00000000, pinned-vs-repicked 0.00000000

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_hep_temporal_clipped_objective_gate` with loss `3.58667445` and ordinary-logit delta `0.00049138`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_hep_temporal_clipped_objective_gate` with loss improvement `0.00006223` and ordinary-logit delta `0.00049138`
