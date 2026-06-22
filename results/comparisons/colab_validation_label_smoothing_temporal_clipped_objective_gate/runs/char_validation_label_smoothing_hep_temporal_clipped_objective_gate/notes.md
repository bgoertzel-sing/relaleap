# char_validation_label_smoothing_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.59249067`
- Residual objective: `supervised_ce_label_smoothing`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `3.5924906730651855`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5869479179382324`, max ordinary-logit delta `0.0`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.586932420730591`, max ordinary-logit delta `0.00012195110321044922`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `0.5`: loss `3.586916923522949`, max ordinary-logit delta `0.00024390220642089844`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
- alpha `1.0`: loss `3.586885929107666`, max ordinary-logit delta `0.0004878044128417969`, support-change fraction `0.0`, pinned-vs-repicked delta `0.0`
