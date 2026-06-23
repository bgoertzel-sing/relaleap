# char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.477319`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `3.4773190021514893`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.4773190021514893`, max ordinary-logit delta `0.0`, support-change fraction `0.375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.4773077964782715`, max ordinary-logit delta `0.00011992454528808594`, support-change fraction `0.375`, pinned-vs-repicked delta `0.0008514225482940674`
- alpha `0.5`: loss `3.477295398712158`, max ordinary-logit delta `0.00024044513702392578`, support-change fraction `0.375`, pinned-vs-repicked delta `0.0017031729221343994`
- alpha `1.0`: loss `3.477271795272827`, max ordinary-logit delta `0.00048100948333740234`, support-change fraction `0.375`, pinned-vs-repicked delta `0.0034065842628479004`
