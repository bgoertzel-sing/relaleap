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
| char_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.39831114 | -0.31763148 | 0.91452196 | 0.01000000 | 0.00000000 | 0.00000000 |
| char_larger_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.15306783 | -0.56287479 | 0.84852436 | 0.01000000 | 0.52148438 | 0.00060484 |
| token_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 4.05925083 | -0.10438013 | 0.97493050 | 0.01000000 | 0.00000000 | 0.00000000 |
| token_larger_support_wide_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 3.52541637 | -0.63821459 | 0.84671682 | 0.01000000 | 0.62109375 | 0.00064898 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/runs/char_larger_hep_temporal_clipped_objective_gate`
- `char_larger_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/runs/char_larger_support_wide_hep_temporal_clipped_objective_gate`
- `token_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/runs/token_larger_hep_temporal_clipped_objective_gate`
- `token_larger_support_wide_hep_temporal_clipped_objective_gate`: `results/comparisons/colab_support_width_larger_char_token_temporal_clipped_objective_gate/runs/token_larger_support_wide_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.39831114, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.39830112, delta 0.00009918, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 3.39829135, delta 0.00019979, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 3.39827180, delta 0.00039887, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `char_larger_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.15306783, delta 0.00000000, support-change 0.52148438, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.15306616, delta 0.00011997, support-change 0.52148438, pinned-vs-repicked 0.00015017, alpha 0.5: loss 3.15306425, delta 0.00023966, support-change 0.52148438, pinned-vs-repicked 0.00030139, alpha 1.0: loss 3.15306067, delta 0.00047971, support-change 0.52148438, pinned-vs-repicked 0.00060484
- `token_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 4.05925083, delta 0.00000000, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.05924129, delta 0.00016761, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 0.5: loss 4.05923080, delta 0.00033515, support-change 0.00000000, pinned-vs-repicked 0.00000000, alpha 1.0: loss 4.05921125, delta 0.00067068, support-change 0.00000000, pinned-vs-repicked 0.00000000
- `token_larger_support_wide_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.52541637, delta 0.00000000, support-change 0.61718750, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.52542686, delta 0.00022793, support-change 0.61718750, pinned-vs-repicked 0.00016159, alpha 0.5: loss 3.52543664, delta 0.00045490, support-change 0.61718750, pinned-vs-repicked 0.00032359, alpha 1.0: loss 3.52545691, delta 0.00091004, support-change 0.62109375, pinned-vs-repicked 0.00064898

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate` with loss `3.15306067` and ordinary-logit delta `0.00047971`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_support_wide_hep_temporal_clipped_objective_gate` with loss improvement `0.00000715` and ordinary-logit delta `0.00047971`
