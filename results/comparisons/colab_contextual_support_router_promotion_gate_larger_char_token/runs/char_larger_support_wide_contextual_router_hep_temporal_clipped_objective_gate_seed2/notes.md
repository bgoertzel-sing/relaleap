# char_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `1.84802544`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.6586458683013916`
- Residual training steps: `50`
- Residual final loss: `1.8480254411697388`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `1.8480254411697388`, max ordinary-logit delta `0.0`, support-change fraction `0.255859375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `1.848086953163147`, max ordinary-logit delta `0.0002951622009277344`, support-change fraction `0.255859375`, pinned-vs-repicked delta `3.5762786865234375e-07`
- alpha `0.5`: loss `1.848148226737976`, max ordinary-logit delta `0.000590205192565918`, support-change fraction `0.255859375`, pinned-vs-repicked delta `4.76837158203125e-07`
- alpha `1.0`: loss `1.8482712507247925`, max ordinary-logit delta `0.001180201768875122`, support-change fraction `0.255859375`, pinned-vs-repicked delta `7.152557373046875e-07`
