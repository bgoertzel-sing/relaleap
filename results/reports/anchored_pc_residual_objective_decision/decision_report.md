# Anchored PC Residual Objective Decision

- Status: `pass`
- Decision: `stop_pc_residual_objective_validation`
- Continue PC residual objective validation: `False`
- Selected PC variant: `None`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean PC minus supervised best HEP loss: `0.00922632`
- Mean anchored PC minus supervised best HEP loss: `0.00005269`
- Mean PC-to-anchored gap reduction: `0.00917363`

## Rationale

The CE-anchored PC objective closes much of the unanchored PC supervised-CE HEP loss gap, but it still does not beat supervised CE residual training in the checked local and Colab artifacts. PC objective validation should stop under the current gate.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | PC best HEP loss | Anchored PC best HEP loss | Anchored minus supervised | Gap reduction | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.59590077 | 3.58672714 | 0.00005269 | 0.00917363 | `results/comparisons/validation_pc_anchor_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.59590077 | 3.58672714 | 0.00005269 | 0.00917363 | `results/comparisons/colab_validation_pc_anchor_temporal_clipped_objective_gate` |

## Next Step

stop PC residual-objective validation under the current gate and select a non-PC residual-learning variant to test next
