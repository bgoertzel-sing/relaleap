# token_larger_hep_support_stress_temporal_clipped

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `4.53275919`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.163630962371826`
- Residual training steps: `50`
- Residual final loss: `4.532759189605713`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.532759189605713`, max ordinary-logit delta `0.0`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.5327324867248535`, max ordinary-logit delta `0.0004354417324066162`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.0009942054748535156`
- alpha `0.5`: loss `4.532703876495361`, max ordinary-logit delta `0.0008707344532012939`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.0019865036010742188`
- alpha `1.0`: loss `4.53264856338501`, max ordinary-logit delta `0.0017414987087249756`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.003972291946411133`
