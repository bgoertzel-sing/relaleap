# char_xlarge_focal_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.0580554`
- Residual objective: `supervised_ce_focal`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.65675950050354`
- Residual training steps: `60`
- Residual final loss: `3.0580554008483887`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.3171539306640625`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.3171510696411133`, max ordinary-logit delta `0.00011309981346130371`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.3171470165252686`, max ordinary-logit delta `0.0002263188362121582`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.3171398639678955`, max ordinary-logit delta `0.00045245885848999023`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
