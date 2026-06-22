# Confidence-Penalty Residual Objective Decision

- Status: `pass`
- Decision: `stop_confidence_penalty_residual_objective_validation`
- Continue confidence-penalty validation: `False`
- Selected variant: `None`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean confidence-penalty minus supervised best HEP loss: `0.00004196`
- Mean confidence-penalty minus supervised final residual loss: `-0.03349614`

## Rationale

The confidence-penalty objective improves its own residual training loss but does not beat supervised CE on best temporal-clipped HEP supervised loss in the checked local and Colab artifacts. It should not continue under the current objective gate.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | Confidence-penalty best HEP loss | Confidence minus supervised | Confidence final residual loss | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.58671641 | 0.00004196 | 3.55324054 | `results/comparisons/validation_confidence_penalty_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.58671641 | 0.00004196 | 3.55324054 | `results/comparisons/colab_validation_confidence_penalty_temporal_clipped_objective_gate` |

## Next Step

select the next non-PC residual objective variant to test under the objective gate
