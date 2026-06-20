# char_validation_hep_support_stress_temporal_clipped

RelaLeap char-level Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `4.60237598`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `4.6023759841918945`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.6023759841918945`, max ordinary-logit delta `0.0`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.602313041687012`, max ordinary-logit delta `0.0006028413772583008`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0009469985961914062`
- alpha `0.5`: loss `4.602249622344971`, max ordinary-logit delta `0.0012055933475494385`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0018916130065917969`
- alpha `1.0`: loss `4.602123260498047`, max ordinary-logit delta `0.0024109184741973877`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0037813186645507812`
