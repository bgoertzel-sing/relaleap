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
| char_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.15306783 | -0.56287479 | 0.84852436 | 0.01000000 | 0.52148438 | 0.00060484 |
| char_larger_capacity_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.15563512 | -0.56030750 | 0.84921524 | 0.01000000 | 0.46289062 | 0.00314734 |
| token_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 3.52541637 | -0.63821459 | 0.84671682 | 0.01000000 | 0.62109375 | 0.00064898 |
| token_larger_capacity_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 3.52521873 | -0.63841223 | 0.84666935 | 0.01000000 | 0.62890625 | 0.00067943 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_post_support_width_capacity_larger_token_objective_gate/runs/char_larger_hep_temporal_clipped_objective_gate`
- `char_larger_capacity_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_post_support_width_capacity_larger_token_objective_gate/runs/char_larger_capacity_hep_temporal_clipped_objective_gate`
- `token_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_post_support_width_capacity_larger_token_objective_gate/runs/token_larger_hep_temporal_clipped_objective_gate`
- `token_larger_capacity_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_post_support_width_capacity_larger_token_objective_gate/runs/token_larger_capacity_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.15306783, delta 0.00000000, support-change 0.52148438, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.15306616, delta 0.00011997, support-change 0.52148438, pinned-vs-repicked 0.00015017, alpha 0.5: loss 3.15306425, delta 0.00023966, support-change 0.52148438, pinned-vs-repicked 0.00030139, alpha 1.0: loss 3.15306067, delta 0.00047971, support-change 0.52148438, pinned-vs-repicked 0.00060484
- `char_larger_capacity_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.15563512, delta 0.00000000, support-change 0.46289062, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.15563297, delta 0.00011918, support-change 0.46289062, pinned-vs-repicked 0.00078270, alpha 0.5: loss 3.15563107, delta 0.00023824, support-change 0.46289062, pinned-vs-repicked 0.00156820, alpha 1.0: loss 3.15562701, delta 0.00047613, support-change 0.46289062, pinned-vs-repicked 0.00314734
- `token_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.52541637, delta 0.00000000, support-change 0.61718750, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.52542686, delta 0.00022793, support-change 0.61718750, pinned-vs-repicked 0.00016159, alpha 0.5: loss 3.52543664, delta 0.00045490, support-change 0.61718750, pinned-vs-repicked 0.00032359, alpha 1.0: loss 3.52545691, delta 0.00091004, support-change 0.62109375, pinned-vs-repicked 0.00064898
- `token_larger_capacity_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.52521873, delta 0.00000000, support-change 0.62890625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.52522898, delta 0.00022650, support-change 0.62890625, pinned-vs-repicked 0.00016922, alpha 0.5: loss 3.52523899, delta 0.00045347, support-change 0.62890625, pinned-vs-repicked 0.00033867, alpha 1.0: loss 3.52525902, delta 0.00090718, support-change 0.62890625, pinned-vs-repicked 0.00067943

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_hep_temporal_clipped_objective_gate` with loss `3.15306067` and ordinary-logit delta `0.00047971`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_hep_temporal_clipped_objective_gate` with loss improvement `0.00000715` and ordinary-logit delta `0.00047971`
