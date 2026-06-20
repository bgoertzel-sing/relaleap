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
| char_larger_hep_support_stress_clipped | supervised_ce | False | True | ok | 3.71594262 | 3.92103243 | 0.20508981 | 1.05519187 | 0.01000000 | 0.65039062 | 0.00319910 |
| char_larger_hep_support_stress_entropy_clipped | supervised_ce | False | True | ok | 3.71594262 | 3.92103243 | 0.20508981 | 1.05519187 | 0.01000000 | 0.65039062 | 0.00319910 |
| char_larger_hep_support_stress_temporal_clipped | supervised_ce | False | True | ok | 3.71594262 | 3.92103243 | 0.20508981 | 1.05519187 | 0.01000000 | 0.65039062 | 0.00319910 |
| char_larger_hep_support_stress_guided_clipped | supervised_ce | False | True | ok | 3.71594262 | 3.92103243 | 0.20508981 | 1.05519187 | 0.01000000 | 0.65039062 | 0.00319910 |

## Artifacts

- `char_larger_hep_support_stress_clipped`: `results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_clipped`
- `char_larger_hep_support_stress_entropy_clipped`: `results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_entropy_clipped`
- `char_larger_hep_support_stress_temporal_clipped`: `results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_temporal_clipped`
- `char_larger_hep_support_stress_guided_clipped`: `results/comparisons/colab_larger_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_larger_hep_support_stress_guided_clipped`

## HEP Alpha Sweeps

- `char_larger_hep_support_stress_clipped`: alpha 0.0: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00080085, alpha 0.5: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00159979, alpha 1.0: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00319910
- `char_larger_hep_support_stress_entropy_clipped`: alpha 0.0: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.92105222, delta 0.00022864, support-change 0.65039062, pinned-vs-repicked 0.00080085, alpha 0.5: loss 3.92107105, delta 0.00045633, support-change 0.65039062, pinned-vs-repicked 0.00159979, alpha 1.0: loss 3.92110944, delta 0.00091171, support-change 0.65039062, pinned-vs-repicked 0.00319910
- `char_larger_hep_support_stress_temporal_clipped`: alpha 0.0: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.92101884, delta 0.00015724, support-change 0.65039062, pinned-vs-repicked 0.00080085, alpha 0.5: loss 3.92100549, delta 0.00031438, support-change 0.65039062, pinned-vs-repicked 0.00159979, alpha 1.0: loss 3.92097831, delta 0.00062868, support-change 0.65039062, pinned-vs-repicked 0.00319910
- `char_larger_hep_support_stress_guided_clipped`: alpha 0.0: loss 3.92103243, delta 0.00000000, support-change 0.65039062, pinned-vs-repicked 0.00000000, alpha 0.25: loss 3.92052722, delta 0.00062227, support-change 0.65039062, pinned-vs-repicked 0.00080085, alpha 0.5: loss 3.92002153, delta 0.00124454, support-change 0.65039062, pinned-vs-repicked 0.00159979, alpha 1.0: loss 3.91901088, delta 0.00248969, support-change 0.65039062, pinned-vs-repicked 0.00319910

## Verdict

- Best HEP alpha by loss: `1.0` in `char_larger_hep_support_stress_guided_clipped` with loss `3.91901088` and ordinary-logit delta `0.00248969`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_larger_hep_support_stress_guided_clipped` with loss improvement `0.00202155` and ordinary-logit delta `0.00248969`
