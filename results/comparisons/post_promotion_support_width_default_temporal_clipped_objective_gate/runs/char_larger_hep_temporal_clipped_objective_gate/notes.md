# char_larger_hep_temporal_clipped_objective_gate

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.1774652`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `3.715942621231079`
- Residual training steps: `50`
- Residual final loss: `3.1774652004241943`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.1774652004241943`, max ordinary-logit delta `0.0`, support-change fraction `0.544921875`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.177461624145508`, max ordinary-logit delta `0.00012023746967315674`, support-change fraction `0.544921875`, pinned-vs-repicked delta `0.0006093680858612061`
- alpha `0.5`: loss `3.177457571029663`, max ordinary-logit delta `0.0002403855323791504`, support-change fraction `0.544921875`, pinned-vs-repicked delta `0.0012208223342895508`
- alpha `1.0`: loss `3.177449941635132`, max ordinary-logit delta `0.0004808008670806885`, support-change fraction `0.546875`, pinned-vs-repicked delta `0.002450287342071533`
