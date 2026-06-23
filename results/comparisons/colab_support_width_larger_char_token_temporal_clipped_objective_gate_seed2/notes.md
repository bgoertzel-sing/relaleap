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
| char_larger_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864587 | 3.36594748 | -0.29269839 | 0.91999816 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864587 | 3.20794559 | -0.45070028 | 0.87681227 | 0.01000000 | 0.28320312 | 0.00028951 |
| token_larger_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 4.23026133 | 4.14451742 | -0.08574391 | 0.97973082 | 0.01000000 | 0.00000000 | 0.00000000 |
| token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 4.23026133 | 3.58100390 | -0.64925743 | 0.84652073 | 0.01000000 | 0.61718750 | 0.01202822 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/char_larger_hep_temporal_clipped_objective_gate_seed2`
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/token_larger_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate_seed2/runs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.36594748, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.36594224, delta 0.00008410, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.36593652, delta 0.00016826, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.36592674, delta 0.00033659, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.20794559, delta 0.00000000, support-change 0.28320312, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.20794654, delta 0.00011444, support-change 0.28320312, pinned-vs-repicked 0.00007227, alpha 0.5: loss 3.20794749, delta 0.00022793, support-change 0.28320312, pinned-vs-repicked 0.00014451, alpha 1.0: loss 3.20794940, delta 0.00045705, support-change 0.28320312, pinned-vs-repicked 0.00028951
- `token_larger_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 4.14451742, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.14450169, delta 0.00022990, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.14448500, delta 0.00045943, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.14445305, delta 0.00091887, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.58100390, delta 0.00000000, support-change 0.61328125, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.58101559, delta 0.00033915, support-change 0.61328125, pinned-vs-repicked 0.00300944, alpha 0.5: loss 3.58102822, delta 0.00067849, support-change 0.61328125, pinned-vs-repicked 0.00601768, alpha 1.0: loss 3.58105326, delta 0.00135659, support-change 0.61718750, pinned-vs-repicked 0.01202822

## Verdict

- Best HEP alpha by loss: `0.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2` with loss `3.20794559` and ordinary-logit delta `0.00000000`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_hep_temporal_clipped_objective_gate_seed2` with loss improvement `0.00002074` and ordinary-logit delta `0.00033659`
