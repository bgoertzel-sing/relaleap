# char_xxlarge_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.38818288`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.760733127593994`
- Residual training steps: `70`
- Residual final loss: `3.3881828784942627`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.3881828784942627`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.3881771564483643`, max ordinary-logit delta `0.00010466575622558594`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.388171672821045`, max ordinary-logit delta `0.00020897388458251953`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.3881609439849854`, max ordinary-logit delta `0.00041747093200683594`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
