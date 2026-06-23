# char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.49417019`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `3.4941701889038086`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.4941701889038086`, max ordinary-logit delta `0.0`, support-change fraction `0.40625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.494156837463379`, max ordinary-logit delta `0.00012123584747314453`, support-change fraction `0.40625`, pinned-vs-repicked delta `0.0016303658485412598`
- alpha `0.5`: loss `3.494143486022949`, max ordinary-logit delta `0.00024259090423583984`, support-change fraction `0.40625`, pinned-vs-repicked delta `0.003261089324951172`
- alpha `1.0`: loss `3.49411678314209`, max ordinary-logit delta `0.00048482418060302734`, support-change fraction `0.40625`, pinned-vs-repicked delta `0.006523221731185913`
