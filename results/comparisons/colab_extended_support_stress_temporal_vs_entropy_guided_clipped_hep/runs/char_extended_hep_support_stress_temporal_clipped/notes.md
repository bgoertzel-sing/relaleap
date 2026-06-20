# char_extended_hep_support_stress_temporal_clipped

RelaLeap char-level Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `4.63250589`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.721653461456299`
- Residual training steps: `30`
- Residual final loss: `4.632505893707275`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `4.632505893707275`, max ordinary-logit delta `0.0`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `4.632448673248291`, max ordinary-logit delta `0.0005948245525360107`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.0013918876647949219`
- alpha `0.5`: loss `4.632391452789307`, max ordinary-logit delta `0.001189887523651123`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.0027818679809570312`
- alpha `1.0`: loss `4.632279396057129`, max ordinary-logit delta `0.002379506826400757`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.00556182861328125`
