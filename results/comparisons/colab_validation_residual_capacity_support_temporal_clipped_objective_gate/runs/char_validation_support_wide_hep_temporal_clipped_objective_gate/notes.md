# char_validation_support_wide_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.49390101`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.7288691997528076`
- Residual training steps: `25`
- Residual final loss: `3.493901014328003`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.493901014328003`, max ordinary-logit delta `0.0`, support-change fraction `0.30859375`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.4938879013061523`, max ordinary-logit delta `0.00012052059173583984`, support-change fraction `0.30859375`, pinned-vs-repicked delta `0.0020474791526794434`
- alpha `0.5`: loss `3.4938747882843018`, max ordinary-logit delta `0.0002416372299194336`, support-change fraction `0.30859375`, pinned-vs-repicked delta `0.00409543514251709`
- alpha `1.0`: loss `3.4938478469848633`, max ordinary-logit delta `0.0004826784133911133`, support-change fraction `0.30859375`, pinned-vs-repicked delta `0.008192360401153564`
