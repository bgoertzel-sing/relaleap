# char_xlarge_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.3177712`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.65675950050354`
- Residual training steps: `60`
- Residual final loss: `3.3177711963653564`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.3177711963653564`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.317767858505249`, max ordinary-logit delta `0.00010979175567626953`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.317763566970825`, max ordinary-logit delta `0.0002193152904510498`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.3177571296691895`, max ordinary-logit delta `0.0004382133483886719`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
