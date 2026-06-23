# char_validation_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.58673739`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `3.5867373943328857`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5867373943328857`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.586721658706665`, max ordinary-logit delta `0.0001227855682373047`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.5867059230804443`, max ordinary-logit delta `0.0002459287643432617`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.5866751670837402`, max ordinary-logit delta `0.0004912614822387695`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
