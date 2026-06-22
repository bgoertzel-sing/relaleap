# char_larger_hep_support_stress_guided_clipped

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.92103243`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.715942621231079`
- Residual training steps: `50`
- Residual final loss: `3.921032428741455`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.921032428741455`, max ordinary-logit delta `0.0`, support-change fraction `0.650390625`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.920527219772339`, max ordinary-logit delta `0.0006222724914550781`, support-change fraction `0.650390625`, pinned-vs-repicked delta `0.0008008480072021484`
- alpha `0.5`: loss `3.9200215339660645`, max ordinary-logit delta `0.0012445449829101562`, support-change fraction `0.650390625`, pinned-vs-repicked delta `0.0015997886657714844`
- alpha `1.0`: loss `3.919010877609253`, max ordinary-logit delta `0.0024896860122680664`, support-change fraction `0.650390625`, pinned-vs-repicked delta `0.0031991004943847656`
