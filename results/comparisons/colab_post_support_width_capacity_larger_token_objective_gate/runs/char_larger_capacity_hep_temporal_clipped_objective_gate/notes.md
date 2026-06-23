# char_larger_capacity_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.15563512`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.715942621231079`
- Residual training steps: `50`
- Residual final loss: `3.155635118484497`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.155635118484497`, max ordinary-logit delta `0.0`, support-change fraction `0.462890625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.155632972717285`, max ordinary-logit delta `0.00011917948722839355`, support-change fraction `0.462890625`, pinned-vs-repicked delta `0.0007827021181583405`
- alpha `0.5`: loss `3.1556310653686523`, max ordinary-logit delta `0.00023823976516723633`, support-change fraction `0.462890625`, pinned-vs-repicked delta `0.0015682019293308258`
- alpha `1.0`: loss `3.1556270122528076`, max ordinary-logit delta `0.00047612935304641724`, support-change fraction `0.462890625`, pinned-vs-repicked delta `0.003147341310977936`
