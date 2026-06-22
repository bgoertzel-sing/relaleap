# char_xxlarge_focal_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.10445833`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7072229385375977`
- Residual training steps: `70`
- Residual final loss: `3.1044583320617676`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.356508255004883`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.3565006256103516`, max ordinary-logit delta `0.00011444091796875`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.3564929962158203`, max ordinary-logit delta `0.0002288818359375`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.356477975845337`, max ordinary-logit delta `0.00045800209045410156`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
