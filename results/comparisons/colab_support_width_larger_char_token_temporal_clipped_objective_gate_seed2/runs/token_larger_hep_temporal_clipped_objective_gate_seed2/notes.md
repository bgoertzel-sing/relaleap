# token_larger_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `4.14451742`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.230261325836182`
- Residual training steps: `50`
- Residual final loss: `4.144517421722412`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.144517421722412`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.144501686096191`, max ordinary-logit delta `0.00022989511489868164`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `4.144484996795654`, max ordinary-logit delta `0.00045943260192871094`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `4.144453048706055`, max ordinary-logit delta `0.0009188652038574219`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
