# char_extended_hep_support_stress_guided_clipped

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
- alpha `0.25`: loss `4.631748676300049`, max ordinary-logit delta `0.0008594989776611328`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.0013918876647949219`
- alpha `0.5`: loss `4.630990982055664`, max ordinary-logit delta `0.001718759536743164`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.0027818679809570312`
- alpha `1.0`: loss `4.629477024078369`, max ordinary-logit delta `0.003437519073486328`, support-change fraction `0.6380208333333334`, pinned-vs-repicked delta `0.00556182861328125`
