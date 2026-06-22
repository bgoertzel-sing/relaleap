# char_xxlarge_focal_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.13634062`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.760733127593994`
- Residual training steps: `70`
- Residual final loss: `3.136340618133545`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.3874692916870117`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.3874640464782715`, max ordinary-logit delta `0.00010776519775390625`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.387458086013794`, max ordinary-logit delta `0.0002161264419555664`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.3874473571777344`, max ordinary-logit delta `0.00043213367462158203`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
