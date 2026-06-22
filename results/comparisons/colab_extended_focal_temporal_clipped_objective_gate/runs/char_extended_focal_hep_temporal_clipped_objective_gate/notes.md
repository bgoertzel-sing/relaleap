# char_extended_focal_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.35723019`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.721653461456299`
- Residual training steps: `30`
- Residual final loss: `3.3572301864624023`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.568485736846924`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5684702396392822`, max ordinary-logit delta `0.00012004375457763672`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.5684547424316406`, max ordinary-logit delta `0.00023984909057617188`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.568423271179199`, max ordinary-logit delta `0.00047898292541503906`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
