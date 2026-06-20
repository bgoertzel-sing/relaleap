# Char Smoke Objective Comparison

Command-driven comparison of Phase 0 char-smoke residual objectives.

- Status: `ok`
- Verdict: `pass`
- Phase 0 invariants: `16` checked, passed `True`
- Artifact invariants: `12` checked, passed `True`
- HEP alpha acceptance: `accepted`
- Loss scale note: Residual objectives may use different loss scales; compare each trajectory against its own initial loss.

## Runs

| Experiment | Objective | Pinned | Stress | Status | Initial loss | Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| char_extended_hep_support_stress_clipped | supervised_ce | False | True | ok | 3.72165346 | 4.63250589 | 0.91085243 | 1.24474402 | 0.01000000 | 0.63802083 | 0.00556183 |
| char_extended_hep_support_stress_entropy_clipped | supervised_ce | False | True | ok | 3.72165346 | 4.63250589 | 0.91085243 | 1.24474402 | 0.01000000 | 0.63802083 | 0.00556183 |
| char_extended_hep_support_stress_temporal_clipped | supervised_ce | False | True | ok | 3.72165346 | 4.63250589 | 0.91085243 | 1.24474402 | 0.01000000 | 0.63802083 | 0.00556183 |
| char_extended_hep_support_stress_guided_clipped | supervised_ce | False | True | ok | 3.72165346 | 4.63250589 | 0.91085243 | 1.24474402 | 0.01000000 | 0.63802083 | 0.00556183 |

## Artifacts

- `char_extended_hep_support_stress_clipped`: `results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_clipped`
- `char_extended_hep_support_stress_entropy_clipped`: `results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_entropy_clipped`
- `char_extended_hep_support_stress_temporal_clipped`: `results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_temporal_clipped`
- `char_extended_hep_support_stress_guided_clipped`: `results/comparisons/colab_extended_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_extended_hep_support_stress_guided_clipped`

## HEP Alpha Sweeps

- `char_extended_hep_support_stress_clipped`: alpha 0.0: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00139189, alpha 0.5: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00278187, alpha 1.0: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00556183
- `char_extended_hep_support_stress_entropy_clipped`: alpha 0.0: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.63263083, delta 0.00063658, support-change 0.63802083, pinned-vs-repicked 0.00139189, alpha 0.5: loss 4.63275480, delta 0.00127077, support-change 0.63802083, pinned-vs-repicked 0.00278187, alpha 1.0: loss 4.63300562, delta 0.00254011, support-change 0.63802083, pinned-vs-repicked 0.00556183
- `char_extended_hep_support_stress_temporal_clipped`: alpha 0.0: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.63244867, delta 0.00059482, support-change 0.63802083, pinned-vs-repicked 0.00139189, alpha 0.5: loss 4.63239145, delta 0.00118989, support-change 0.63802083, pinned-vs-repicked 0.00278187, alpha 1.0: loss 4.63227940, delta 0.00237951, support-change 0.63802083, pinned-vs-repicked 0.00556183
- `char_extended_hep_support_stress_guided_clipped`: alpha 0.0: loss 4.63250589, delta 0.00000000, support-change 0.63802083, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.63174868, delta 0.00085950, support-change 0.63802083, pinned-vs-repicked 0.00139189, alpha 0.5: loss 4.63099098, delta 0.00171876, support-change 0.63802083, pinned-vs-repicked 0.00278187, alpha 1.0: loss 4.62947702, delta 0.00343752, support-change 0.63802083, pinned-vs-repicked 0.00556183

## Verdict

- Best HEP alpha by loss: `1.0` in `char_extended_hep_support_stress_guided_clipped` with loss `4.62947702` and ordinary-logit delta `0.00343752`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_extended_hep_support_stress_guided_clipped` with loss improvement `0.00302887` and ordinary-logit delta `0.00343752`
