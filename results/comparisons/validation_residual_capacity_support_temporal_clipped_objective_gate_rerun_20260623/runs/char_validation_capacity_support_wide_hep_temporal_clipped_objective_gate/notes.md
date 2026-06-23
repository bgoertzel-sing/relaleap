# char_validation_capacity_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.48526955`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.728869676589966`
- Residual training steps: `25`
- Residual final loss: `3.485269546508789`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.485269546508789`, max ordinary-logit delta `0.0`, support-change fraction `0.28515625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.485257625579834`, max ordinary-logit delta `0.00012112408876419067`, support-change fraction `0.28515625`, pinned-vs-repicked delta `0.0003673732280731201`
- alpha `0.5`: loss `3.4852454662323`, max ordinary-logit delta `0.00024247914552688599`, support-change fraction `0.28515625`, pinned-vs-repicked delta `0.0007347017526626587`
- alpha `1.0`: loss `3.4852213859558105`, max ordinary-logit delta `0.00048495829105377197`, support-change fraction `0.2890625`, pinned-vs-repicked delta `0.0014709681272506714`
