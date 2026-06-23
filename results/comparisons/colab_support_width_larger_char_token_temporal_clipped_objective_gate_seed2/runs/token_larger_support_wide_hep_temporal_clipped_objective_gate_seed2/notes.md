# token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cuda`
- CUDA available: `True`
- Final smoke loss: `3.5810039`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.230261325836182`
- Residual training steps: `50`
- Residual final loss: `3.5810039043426514`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5810039043426514`, max ordinary-logit delta `0.0`, support-change fraction `0.61328125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5810155868530273`, max ordinary-logit delta `0.00033915042877197266`, support-change fraction `0.61328125`, pinned-vs-repicked delta `0.0030094385147094727`
- alpha `0.5`: loss `3.5810282230377197`, max ordinary-logit delta `0.0006784945726394653`, support-change fraction `0.61328125`, pinned-vs-repicked delta `0.0060176849365234375`
- alpha `1.0`: loss `3.5810532569885254`, max ordinary-logit delta `0.0013565942645072937`, support-change fraction `0.6171875`, pinned-vs-repicked delta `0.012028217315673828`
