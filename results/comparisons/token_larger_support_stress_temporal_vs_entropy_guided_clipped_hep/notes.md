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
| token_larger_hep_support_stress_clipped | supervised_ce | False | True | ok | 4.16363096 | 4.53275919 | 0.36912823 | 1.08865537 | 0.01000000 | 0.60546875 | 0.00397229 |
| token_larger_hep_support_stress_entropy_clipped | supervised_ce | False | True | ok | 4.16363096 | 4.53275919 | 0.36912823 | 1.08865537 | 0.01000000 | 0.60546875 | 0.00397229 |
| token_larger_hep_support_stress_temporal_clipped | supervised_ce | False | True | ok | 4.16363096 | 4.53275919 | 0.36912823 | 1.08865537 | 0.01000000 | 0.60546875 | 0.00397229 |
| token_larger_hep_support_stress_guided_clipped | supervised_ce | False | True | ok | 4.16363096 | 4.53275919 | 0.36912823 | 1.08865537 | 0.01000000 | 0.60546875 | 0.00397229 |

## Artifacts

- `token_larger_hep_support_stress_clipped`: `results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_clipped`
- `token_larger_hep_support_stress_entropy_clipped`: `results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_entropy_clipped`
- `token_larger_hep_support_stress_temporal_clipped`: `results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_temporal_clipped`
- `token_larger_hep_support_stress_guided_clipped`: `results/comparisons/token_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/token_larger_hep_support_stress_guided_clipped`

## HEP Alpha Sweeps

- `token_larger_hep_support_stress_clipped`: alpha 0.0: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00099421, alpha 0.5: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00198650, alpha 1.0: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00397229
- `token_larger_hep_support_stress_entropy_clipped`: alpha 0.0: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.53281307, delta 0.00075245, support-change 0.60546875, pinned-vs-repicked 0.00099421, alpha 0.5: loss 4.53286791, delta 0.00150442, support-change 0.60546875, pinned-vs-repicked 0.00198650, alpha 1.0: loss 4.53297710, delta 0.00301218, support-change 0.60546875, pinned-vs-repicked 0.00397229
- `token_larger_hep_support_stress_temporal_clipped`: alpha 0.0: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.53273249, delta 0.00043544, support-change 0.60546875, pinned-vs-repicked 0.00099421, alpha 0.5: loss 4.53270388, delta 0.00087073, support-change 0.60546875, pinned-vs-repicked 0.00198650, alpha 1.0: loss 4.53264856, delta 0.00174150, support-change 0.60546875, pinned-vs-repicked 0.00397229
- `token_larger_hep_support_stress_guided_clipped`: alpha 0.0: loss 4.53275919, delta 0.00000000, support-change 0.60546875, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.53176451, delta 0.00114572, support-change 0.60546875, pinned-vs-repicked 0.00099421, alpha 0.5: loss 4.53076935, delta 0.00229108, support-change 0.60546875, pinned-vs-repicked 0.00198650, alpha 1.0: loss 4.52877951, delta 0.00458205, support-change 0.60546875, pinned-vs-repicked 0.00397229

## Verdict

- Best HEP alpha by loss: `1.0` in `token_larger_hep_support_stress_guided_clipped` with loss `4.52877951` and ordinary-logit delta `0.00458205`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `token_larger_hep_support_stress_guided_clipped` with loss improvement `0.00397968` and ordinary-logit delta `0.00458205`
