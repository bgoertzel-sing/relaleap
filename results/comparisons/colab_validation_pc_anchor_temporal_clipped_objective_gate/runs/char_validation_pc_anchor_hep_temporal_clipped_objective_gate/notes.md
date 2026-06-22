# char_validation_pc_anchor_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `0.3874163`
- Residual objective: `pc_logit_mse_ce_anchor`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `0.3874163031578064`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.586789131164551`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5867738723754883`, max ordinary-logit delta `0.00012230873107910156`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.5867581367492676`, max ordinary-logit delta `0.0002442598342895508`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.5867271423339844`, max ordinary-logit delta `0.0004886388778686523`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
