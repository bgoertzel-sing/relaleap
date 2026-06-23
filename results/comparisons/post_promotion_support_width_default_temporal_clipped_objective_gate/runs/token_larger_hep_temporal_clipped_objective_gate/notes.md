# token_larger_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.48588872`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.163630962371826`
- Residual training steps: `50`
- Residual final loss: `3.485888719558716`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.485888719558716`, max ordinary-logit delta `0.0`, support-change fraction `0.6484375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.4859042167663574`, max ordinary-logit delta `0.00022414326667785645`, support-change fraction `0.6484375`, pinned-vs-repicked delta `0.003247886896133423`
- alpha `0.5`: loss `3.485919713973999`, max ordinary-logit delta `0.0004482269287109375`, support-change fraction `0.6484375`, pinned-vs-repicked delta `0.006502866744995117`
- alpha `1.0`: loss `3.4859507083892822`, max ordinary-logit delta `0.000896155834197998`, support-change fraction `0.6484375`, pinned-vs-repicked delta `0.013034254312515259`
