# char_validation_temporal_consistency_w020_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `4.31279516`
- Residual objective: `supervised_ce_temporal_consistency`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `4.312795162200928`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5867269039154053`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5867116451263428`, max ordinary-logit delta `0.00012302398681640625`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.586695909500122`, max ordinary-logit delta `0.0002454519271850586`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.5866646766662598`, max ordinary-logit delta `0.0004912614822387695`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
