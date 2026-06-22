# char_validation_hep_support_stress_temporal_clipped

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `4.60237598`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `4.6023759841918945`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.6023759841918945`, max ordinary-logit delta `0.0`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.602313041687012`, max ordinary-logit delta `0.0006028413772583008`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0009465217590332031`
- alpha `0.5`: loss `4.602248668670654`, max ordinary-logit delta `0.0012055933475494385`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0018911361694335938`
- alpha `1.0`: loss `4.602122783660889`, max ordinary-logit delta `0.0024109482765197754`, support-change fraction `0.59765625`, pinned-vs-repicked delta `0.0037813186645507812`
