# char_larger_focal_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.15012836`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.715942621231079`
- Residual training steps: `50`
- Residual final loss: `3.1501283645629883`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.3967368602752686`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.3967273235321045`, max ordinary-logit delta `0.00010156631469726562`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.3967175483703613`, max ordinary-logit delta `0.0002033710479736328`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.396697998046875`, max ordinary-logit delta `0.00040531158447265625`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
