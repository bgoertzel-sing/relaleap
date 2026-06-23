# char_validation_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.58673668`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `3.5867366790771484`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5867366790771484`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.586721420288086`, max ordinary-logit delta `0.00012290477752685547`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.586705446243286`, max ordinary-logit delta `0.0002453327178955078`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.586674451828003`, max ordinary-logit delta `0.0004912614822387695`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
