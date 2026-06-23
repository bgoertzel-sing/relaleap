# token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.5973053`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.230261325836182`
- Residual training steps: `50`
- Residual final loss: `3.5973052978515625`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5973052978515625`, max ordinary-logit delta `0.0`, support-change fraction `0.3984375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5973126888275146`, max ordinary-logit delta `0.00022649765014648438`, support-change fraction `0.3984375`, pinned-vs-repicked delta `0.00042891502380371094`
- alpha `0.5`: loss `3.597320079803467`, max ordinary-logit delta `0.0004527568817138672`, support-change fraction `0.3984375`, pinned-vs-repicked delta `0.000859379768371582`
- alpha `1.0`: loss `3.59733510017395`, max ordinary-logit delta `0.0009059906005859375`, support-change fraction `0.3984375`, pinned-vs-repicked delta `0.0017241239547729492`
