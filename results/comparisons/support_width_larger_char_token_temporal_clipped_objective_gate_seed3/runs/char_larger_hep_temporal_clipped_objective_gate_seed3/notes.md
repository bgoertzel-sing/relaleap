# char_larger_hep_temporal_clipped_objective_gate_seed3

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.50802588`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7707438468933105`
- Residual training steps: `50`
- Residual final loss: `3.508025884628296`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.508025884628296`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5080182552337646`, max ordinary-logit delta `0.00017768144607543945`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.5080111026763916`, max ordinary-logit delta `0.0003555119037628174`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.5079963207244873`, max ordinary-logit delta `0.0007106959819793701`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
