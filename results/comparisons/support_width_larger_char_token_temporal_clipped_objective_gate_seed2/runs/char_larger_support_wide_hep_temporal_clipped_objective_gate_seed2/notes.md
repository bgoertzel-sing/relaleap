# char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.14497447`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.6586453914642334`
- Residual training steps: `50`
- Residual final loss: `3.14497447013855`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.14497447013855`, max ordinary-logit delta `0.0`, support-change fraction `0.4453125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.14497447013855`, max ordinary-logit delta `0.00011584162712097168`, support-change fraction `0.4453125`, pinned-vs-repicked delta `0.0006742477416992188`
- alpha `0.5`: loss `3.1449739933013916`, max ordinary-logit delta `0.0002320408821105957`, support-change fraction `0.4453125`, pinned-vs-repicked delta `0.0013501644134521484`
- alpha `1.0`: loss `3.1449732780456543`, max ordinary-logit delta `0.0004639923572540283`, support-change fraction `0.447265625`, pinned-vs-repicked delta `0.002706974744796753`
