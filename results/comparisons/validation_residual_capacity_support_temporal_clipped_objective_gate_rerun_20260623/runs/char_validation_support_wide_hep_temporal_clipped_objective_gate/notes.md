# char_validation_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.48482919`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `3.4848291873931885`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.4848291873931885`, max ordinary-logit delta `0.0`, support-change fraction `0.41015625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.48481822013855`, max ordinary-logit delta `0.00011789798736572266`, support-change fraction `0.41015625`, pinned-vs-repicked delta `0.0008053779602050781`
- alpha `0.5`: loss `3.484806537628174`, max ordinary-logit delta `0.00023674964904785156`, support-change fraction `0.41015625`, pinned-vs-repicked delta `0.001611173152923584`
- alpha `1.0`: loss `3.4847843647003174`, max ordinary-logit delta `0.00047338008880615234`, support-change fraction `0.41015625`, pinned-vs-repicked delta `0.003224387764930725`
