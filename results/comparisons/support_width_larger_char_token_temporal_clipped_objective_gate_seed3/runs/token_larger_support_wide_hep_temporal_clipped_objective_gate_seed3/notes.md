# token_larger_support_wide_hep_temporal_clipped_objective_gate_seed3

RelaLeap Phase 0 smoke run.

- Status: `ok`
- Error: `none`
- Device: `cpu`
- CUDA available: `False`
- Final smoke loss: `3.59574461`
- Residual objective: `supervised_ce`
- Pinned support: `False`
- Support stress: `True`
- Base loss: `4.27324104309082`
- Residual training steps: `50`
- Residual final loss: `3.5957446098327637`

## Invariants

- frozen_base_unchanged: `True`
- hep_alpha_0_equivalence: `True`
- residual_parameters_updated: `True`
- zero_init_identity: `True`

## HEP Alpha Sweep

- alpha `0.0`: loss `3.5957446098327637`, max ordinary-logit delta `0.0`, support-change fraction `0.55078125`, pinned-vs-repicked delta `0.0`
- alpha `0.25`: loss `3.5957577228546143`, max ordinary-logit delta `0.00023221969604492188`, support-change fraction `0.55078125`, pinned-vs-repicked delta `0.003476947546005249`
- alpha `0.5`: loss `3.5957698822021484`, max ordinary-logit delta `0.00046539306640625`, support-change fraction `0.55078125`, pinned-vs-repicked delta `0.006952822208404541`
- alpha `1.0`: loss `3.5957961082458496`, max ordinary-logit delta `0.0009303092956542969`, support-change fraction `0.55078125`, pinned-vs-repicked delta `0.013901114463806152`
