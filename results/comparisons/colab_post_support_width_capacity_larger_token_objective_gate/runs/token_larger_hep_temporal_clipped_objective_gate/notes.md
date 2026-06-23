# token_larger_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.52541637`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.163630962371826`
- Residual training steps: `50`
- Residual final loss: `3.525416374206543`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.525416374206543`, max ordinary-logit delta `0.0`, support-change fraction `0.6171875`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5254268646240234`, max ordinary-logit delta `0.00022792816162109375`, support-change fraction `0.6171875`, pinned-vs-repicked delta `0.00016158819198608398`
- alpha `0.5`: loss `3.5254366397857666`, max ordinary-logit delta `0.00045490264892578125`, support-change fraction `0.6171875`, pinned-vs-repicked delta `0.0003235936164855957`
- alpha `1.0`: loss `3.5254569053649902`, max ordinary-logit delta `0.0009100437164306641`, support-change fraction `0.62109375`, pinned-vs-repicked delta `0.0006489753723144531`
