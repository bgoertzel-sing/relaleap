# char_larger_support_wide_hep_temporal_clipped_objective_gate_seed3

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.10037756`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7707438468933105`
- Residual training steps: `50`
- Residual final loss: `3.1003775596618652`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.1003775596618652`, max ordinary-logit delta `0.0`, support-change fraction `0.6328125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.1003806591033936`, max ordinary-logit delta `0.00018286705017089844`, support-change fraction `0.6328125`, pinned-vs-repicked delta `0.0003350973129272461`
- alpha `0.5`: loss `3.100383758544922`, max ordinary-logit delta `0.0003647804260253906`, support-change fraction `0.6328125`, pinned-vs-repicked delta `0.0006713271141052246`
- alpha `1.0`: loss `3.1003901958465576`, max ordinary-logit delta `0.0007302761077880859`, support-change fraction `0.662109375`, pinned-vs-repicked delta `0.001345515251159668`
