# Label-Smoothing Residual Objective Decision

- Status: `pass`
- Decision: `stop_label_smoothing_residual_objective_validation`
- Continue label-smoothing validation: `False`
- Selected variant: `None`
- Default residual objective: `supervised_ce`
- Backends: `colab, local`
- Mean label-smoothing minus supervised best HEP loss: `0.00021148`
- Mean label-smoothing minus supervised final residual loss: `0.00575399`

## Rationale

The label-smoothing objective improves its own residual training loss but does not beat supervised CE on best temporal-clipped HEP supervised loss in the checked local and Colab artifacts. It should not continue under the current objective gate.

## Evidence

| Backend | Artifact check | Supervised best HEP loss | Label-smoothing best HEP loss | Label smoothing minus supervised | Label smoothing final residual loss | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local | pass | 3.58667445 | 3.58688593 | 0.00021148 | 3.59249067 | `results/comparisons/validation_label_smoothing_temporal_clipped_objective_gate` |
| colab | pass | 3.58667445 | 3.58688593 | 0.00021148 | 3.59249067 | `results/comparisons/colab_validation_label_smoothing_temporal_clipped_objective_gate` |

## Next Step

select the next non-PC residual objective variant to test under the objective gate
