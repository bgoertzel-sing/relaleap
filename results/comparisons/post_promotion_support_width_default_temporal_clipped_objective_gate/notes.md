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
| char_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 3.71594262 | 3.17746520 | -0.53847742 | 0.85508995 | 0.01000000 | 0.54687500 | 0.00245029 |
| token_larger_hep_temporal_clipped_objective_gate | supervised_ce | False | True | ok | 4.16363096 | 3.48588872 | -0.67774224 | 0.83722327 | 0.01000000 | 0.64843750 | 0.01303425 |

## Artifacts

- `char_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/post_promotion_support_width_default_temporal_clipped_objective_gate/runs/char_larger_hep_temporal_clipped_objective_gate`
- `token_larger_hep_temporal_clipped_objective_gate`: `results/comparisons/post_promotion_support_width_default_temporal_clipped_objective_gate/runs/token_larger_hep_temporal_clipped_objective_gate`

## HEP Alpha Sweeps

- `char_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.17746520, delta 0.00000000, support-change 0.54492188, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.17746162, delta 0.00012024, support-change 0.54492188, pinned-vs-repicked 0.00060937, alpha 0.5: loss 3.17745757, delta 0.00024039, support-change 0.54492188, pinned-vs-repicked 0.00122082, alpha 1.0: loss 3.17744994, delta 0.00048080, support-change 0.54687500, pinned-vs-repicked 0.00245029
- `token_larger_hep_temporal_clipped_objective_gate`: alpha 0.0: loss 3.48588872, delta 0.00000000, support-change 0.64843750, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.48590422, delta 0.00022414, support-change 0.64843750, pinned-vs-repicked 0.00324789, alpha 0.5: loss 3.48591971, delta 0.00044823, support-change 0.64843750, pinned-vs-repicked 0.00650287, alpha 1.0: loss 3.48595071, delta 0.00089616, support-change 0.64843750, pinned-vs-repicked 0.01303425

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_hep_temporal_clipped_objective_gate` with loss `3.17744994` and ordinary-logit delta `0.00048080`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_hep_temporal_clipped_objective_gate` with loss improvement `0.00001526` and ordinary-logit delta `0.00048080`
