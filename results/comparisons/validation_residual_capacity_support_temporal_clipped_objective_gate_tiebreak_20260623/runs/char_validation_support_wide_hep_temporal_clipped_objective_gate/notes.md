# char_validation_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.48881698`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `3.488816976547241`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.488816976547241`, max ordinary-logit delta `0.0`, support-change fraction `0.3515625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.4888057708740234`, max ordinary-logit delta `0.00011420249938964844`, support-change fraction `0.3515625`, pinned-vs-repicked delta `0.0007315874099731445`
- alpha `0.5`: loss `3.4887943267822266`, max ordinary-logit delta `0.00022840499877929688`, support-change fraction `0.3515625`, pinned-vs-repicked delta `0.0014630556106567383`
- alpha `1.0`: loss `3.4887712001800537`, max ordinary-logit delta `0.0004563331604003906`, support-change fraction `0.35546875`, pinned-vs-repicked delta `0.00292813777923584`
