# RelaLeap Objective Comparison

Command-driven comparison of Phase 0 residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `16` checked, passed `True`
- Artifact invariants: `12` checked, passed `True`
- HEP alpha acceptance: `no_accepted_alpha`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864587 | 3.20794559 | -0.45070028 | 0.87681227 | 0.01000000 | 0.28320312 | 0.00028951 |
| char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2 | supervised_ce | False | True | ok | 3.65864587 | 1.84802544 | -1.81062043 | 0.50511187 | 0.01000000 | 0.25585938 | 0.00000072 |
| token_larger_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 3.52541637 | -0.63821459 | 0.84671682 | 0.01000000 | 0.62109375 | 0.00064898 |
| token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 2.86810851 | -1.29552245 | 0.68884792 | 0.01000000 | 0.32421875 | 0.00835282 |

## Artifacts

- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_contextual_support_router_promotion_gate_larger_char_token/runs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`
- `char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2`: `results/comparisons/colab_contextual_support_router_promotion_gate_larger_char_token/runs/char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2`
- `token_larger_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_contextual_support_router_promotion_gate_larger_char_token/runs/token_larger_support_wide_hep_temporal_clipped_objective_gate`
- `token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_contextual_support_router_promotion_gate_larger_char_token/runs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 3.20794559, delta 0.00000000, support-change 0.28320312, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.20794654, delta 0.00011444, support-change 0.28320312, pinned-vs-repicked 0.00007227, alpha 0.5: loss 3.20794749, delta 0.00022793, support-change 0.28320312, pinned-vs-repicked 0.00014451, alpha 1.0: loss 3.20794940, delta 0.00045705, support-change 0.28320312, pinned-vs-repicked 0.00028951
- `char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2`: alpha 0.0: loss 1.84802544, delta 0.00000000, support-change 0.25585938, pinned-vs-repicked 0.00000000, alpha 0.25: loss 1.84808695, delta 0.00029516, support-change 0.25585938, pinned-vs-repicked 0.00000036, alpha 0.5: loss 1.84814823, delta 0.00059021, support-change 0.25585938, pinned-vs-repicked 0.00000048, alpha 1.0: loss 1.84827125, delta 0.00118020, support-change 0.25585938, pinned-vs-repicked 0.00000072
- `token_larger_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.52541637, delta 0.00000000, support-change 0.61718750, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.52542686, delta 0.00022793, support-change 0.61718750, pinned-vs-repicked 0.00016159, alpha 0.5: loss 3.52543664, delta 0.00045490, support-change 0.61718750, pinned-vs-repicked 0.00032359, alpha 1.0: loss 3.52545691, delta 0.00091004, support-change 0.62109375, pinned-vs-repicked 0.00064898
- `token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 2.86810851, delta 0.00000000, support-change 0.32031250, pinned-vs-repicked 0.00000000, alpha 0.25: loss 2.86815667, delta 0.00036716, support-change 0.32031250, pinned-vs-repicked 0.00207964, alpha 0.5: loss 2.86820436, delta 0.00073397, support-change 0.32031250, pinned-vs-repicked 0.00416534, alpha 1.0: loss 2.86829996, delta 0.00146699, support-change 0.32421875, pinned-vs-repicked 0.00835282

## Verdict

- Best HEP alpha by loss: `0.0` in `char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2` with loss `1.84802544` and ordinary-logit delta `0.00000000`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `none`
