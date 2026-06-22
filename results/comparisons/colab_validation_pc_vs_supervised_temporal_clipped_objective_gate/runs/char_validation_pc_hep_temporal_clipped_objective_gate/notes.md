# char_validation_pc_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `0.02871708`
- Residual objective: `pc_logit_mse`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `0.028717081993818283`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5959599018096924`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.595945119857788`, max ordinary-logit delta `0.00012123584747314453`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.5959298610687256`, max ordinary-logit delta `0.0002422332763671875`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.595900774002075`, max ordinary-logit delta `0.00048422813415527344`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
