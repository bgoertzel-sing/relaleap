# Margin-Penalty Residual Objective Decision

- Status: `pass`
- Decision: `stop_margin_penalty_residual_objective_validation`
- Continue margin-penalty validation: `False`
- Selected variant: `None`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean margin-penalty minus supervised best HEP loss: `0.00021887`
- Mean margin-penalty minus supervised final residual loss: `0.01426101`

## Rationale

The margin-penalty objective improves its own residual training loss but does not beat supervised CE on best temporal-clipped HEP supervised loss in the checked local and Colab artifacts. It should not continue under the current objective gate.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | Margin-penalty best HEP loss | Margin minus supervised | Margin final residual loss | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.58689332 | 0.00021887 | 3.60099769 | `results/comparisons/validation_margin_penalty_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.58689332 | 0.00021887 | 3.60099769 | `results/comparisons/colab_validation_margin_penalty_temporal_clipped_objective_gate` |

## Next Step

select the next non-PC residual objective variant to test under the objective gate
