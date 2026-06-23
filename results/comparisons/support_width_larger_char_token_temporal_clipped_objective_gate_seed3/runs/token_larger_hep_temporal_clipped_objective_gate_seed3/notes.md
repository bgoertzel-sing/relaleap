# token_larger_hep_temporal_clipped_objective_gate_seed3

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `4.17101955`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.27324104309082`
- Residual training steps: `50`
- Residual final loss: `4.171019554138184`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.171019554138184`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.171008586883545`, max ordinary-logit delta `0.00022029876708984375`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `4.170997619628906`, max ordinary-logit delta `0.00043964385986328125`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `4.170975685119629`, max ordinary-logit delta `0.0008800029754638672`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
