# char_larger_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.15306783`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.715942621231079`
- Residual training steps: `50`
- Residual final loss: `3.1530678272247314`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.1530678272247314`, max ordinary-logit delta `0.0`, support-change fraction `0.521484375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.1530661582946777`, max ordinary-logit delta `0.00011996924877166748`, support-change fraction `0.521484375`, pinned-vs-repicked delta `0.00015017390251159668`
- alpha `0.5`: loss `3.153064250946045`, max ordinary-logit delta `0.00023965537548065186`, support-change fraction `0.521484375`, pinned-vs-repicked delta `0.0003013908863067627`
- alpha `1.0`: loss `3.1530606746673584`, max ordinary-logit delta `0.0004797130823135376`, support-change fraction `0.521484375`, pinned-vs-repicked delta `0.0006048381328582764`
