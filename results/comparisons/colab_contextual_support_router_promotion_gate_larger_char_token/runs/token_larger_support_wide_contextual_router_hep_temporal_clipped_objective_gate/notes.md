# token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `2.86810851`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.163630962371826`
- Residual training steps: `50`
- Residual final loss: `2.8681085109710693`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `2.8681085109710693`, max ordinary-logit delta `0.0`, support-change fraction `0.3203125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `2.868156671524048`, max ordinary-logit delta `0.00036716461181640625`, support-change fraction `0.3203125`, pinned-vs-repicked delta `0.0020796358585357666`
- alpha `0.5`: loss `2.868204355239868`, max ordinary-logit delta `0.0007339715957641602`, support-change fraction `0.3203125`, pinned-vs-repicked delta `0.004165336489677429`
- alpha `1.0`: loss `2.868299961090088`, max ordinary-logit delta `0.001466989517211914`, support-change fraction `0.32421875`, pinned-vs-repicked delta `0.008352816104888916`
