# token_larger_focal_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.99827075`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.230261325836182`
- Residual training steps: `50`
- Residual final loss: `3.9982707500457764`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.145286560058594`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.145269393920898`, max ordinary-logit delta `0.0002459287643432617`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `4.145253658294678`, max ordinary-logit delta `0.0004918575286865234`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `4.14522123336792`, max ordinary-logit delta `0.000983119010925293`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
