# token_larger_capacity_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.52521873`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.163630962371826`
- Residual training steps: `50`
- Residual final loss: `3.5252187252044678`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5252187252044678`, max ordinary-logit delta `0.0`, support-change fraction `0.62890625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.525228977203369`, max ordinary-logit delta `0.00022649765014648438`, support-change fraction `0.62890625`, pinned-vs-repicked delta `0.00016921758651733398`
- alpha `0.5`: loss `3.5252389907836914`, max ordinary-logit delta `0.0004534721374511719`, support-change fraction `0.62890625`, pinned-vs-repicked delta `0.00033867359161376953`
- alpha `1.0`: loss `3.525259017944336`, max ordinary-logit delta `0.0009071826934814453`, support-change fraction `0.62890625`, pinned-vs-repicked delta `0.0006794333457946777`
