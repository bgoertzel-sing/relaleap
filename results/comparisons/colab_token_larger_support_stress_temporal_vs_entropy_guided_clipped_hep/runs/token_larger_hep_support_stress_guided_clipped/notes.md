# token_larger_hep_support_stress_guided_clipped

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
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
- alpha `0.25`: loss `4.531764030456543`, max ordinary-logit delta `0.0011456608772277832`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.0009946823120117188`
- alpha `0.5`: loss `4.530769348144531`, max ordinary-logit delta `0.0022910237312316895`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.001986980438232422`
- alpha `1.0`: loss `4.52877950668335`, max ordinary-logit delta `0.0045819878578186035`, support-change fraction `0.60546875`, pinned-vs-repicked delta `0.003972053527832031`
