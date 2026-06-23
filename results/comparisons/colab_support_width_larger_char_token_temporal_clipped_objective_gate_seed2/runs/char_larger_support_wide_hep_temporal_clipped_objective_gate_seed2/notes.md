# char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.20794559`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.6586458683013916`
- Residual training steps: `50`
- Residual final loss: `3.2079455852508545`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.2079455852508545`, max ordinary-logit delta `0.0`, support-change fraction `0.283203125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.207946538925171`, max ordinary-logit delta `0.00011444091796875`, support-change fraction `0.283203125`, pinned-vs-repicked delta `7.227063179016113e-05`
- alpha `0.5`: loss `3.2079474925994873`, max ordinary-logit delta `0.00022792816162109375`, support-change fraction `0.283203125`, pinned-vs-repicked delta `0.00014451146125793457`
- alpha `1.0`: loss `3.20794939994812`, max ordinary-logit delta `0.0004570484161376953`, support-change fraction `0.283203125`, pinned-vs-repicked delta `0.0002895146608352661`
