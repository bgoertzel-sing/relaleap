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
| char_validation_hep_support_stress_clipped | supervised_ce | False | True | ok | 3.72886920 | 4.60237598 | 0.87350678 | 1.23425514 | 0.01000000 | 0.59765625 | 0.00378132 |
| char_validation_hep_support_stress_entropy_clipped | supervised_ce | False | True | ok | 3.72886920 | 4.60237598 | 0.87350678 | 1.23425514 | 0.01000000 | 0.59765625 | 0.00378132 |
| char_validation_hep_support_stress_temporal_clipped | supervised_ce | False | True | ok | 3.72886920 | 4.60237598 | 0.87350678 | 1.23425514 | 0.01000000 | 0.59765625 | 0.00378132 |
| char_validation_hep_support_stress_guided_clipped | supervised_ce | False | True | ok | 3.72886920 | 4.60237598 | 0.87350678 | 1.23425514 | 0.01000000 | 0.59765625 | 0.00378132 |

## Artifacts

- `char_validation_hep_support_stress_clipped`: `results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_clipped`
- `char_validation_hep_support_stress_entropy_clipped`: `results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_entropy_clipped`
- `char_validation_hep_support_stress_temporal_clipped`: `results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_temporal_clipped`
- `char_validation_hep_support_stress_guided_clipped`: `results/comparisons/colab_validation_support_stress_temporal_vs_entropy_guided_clipped_hep/runs/char_validation_hep_support_stress_guided_clipped`

## HEP Alpha Sweeps

- `char_validation_hep_support_stress_clipped`: alpha 0.0: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00094652, alpha 0.5: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00189114, alpha 1.0: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00378132
- `char_validation_hep_support_stress_entropy_clipped`: alpha 0.0: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.60249758, delta 0.00063944, support-change 0.59765625, pinned-vs-repicked 0.00094652, alpha 0.5: loss 4.60261917, delta 0.00127840, support-change 0.59765625, pinned-vs-repicked 0.00189114, alpha 1.0: loss 4.60286188, delta 0.00255585, support-change 0.59765625, pinned-vs-repicked 0.00378132
- `char_validation_hep_support_stress_temporal_clipped`: alpha 0.0: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.60231304, delta 0.00060284, support-change 0.59765625, pinned-vs-repicked 0.00094652, alpha 0.5: loss 4.60224867, delta 0.00120559, support-change 0.59765625, pinned-vs-repicked 0.00189114, alpha 1.0: loss 4.60212278, delta 0.00241095, support-change 0.59765625, pinned-vs-repicked 0.00378132
- `char_validation_hep_support_stress_guided_clipped`: alpha 0.0: loss 4.60237598, delta 0.00000000, support-change 0.59765625, pinned-vs-repicked 0.00000000, alpha 0.25: loss 4.60161734, delta 0.00086355, support-change 0.59765625, pinned-vs-repicked 0.00094652, alpha 0.5: loss 4.60085821, delta 0.00172704, support-change 0.59765625, pinned-vs-repicked 0.00189114, alpha 1.0: loss 4.59934044, delta 0.00345415, support-change 0.59765625, pinned-vs-repicked 0.00378132

## Verdict

- Best HEP alpha by loss: `1.0` in `char_validation_hep_support_stress_guided_clipped` with loss `4.59934044` and ordinary-logit delta `0.00345415`
- HEP acceptance policy: require nonzero alpha, loss improvement over alpha 0 greater than `0.00000000`, and ordinary-logit delta at or below `0.10000000`
- Accepted HEP alpha: `1.0` in `char_validation_hep_support_stress_guided_clipped` with loss improvement `0.00303555` and ordinary-logit delta `0.00345415`
