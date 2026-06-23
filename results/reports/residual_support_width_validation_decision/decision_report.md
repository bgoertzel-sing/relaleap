# Residual Support Width Validation Decision

- Status: `pass`
- Decision: `continue_residual_support_width_validation`
- Selected direction: `support_width_repeat_or_capacity_interaction_validation`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`

## Rationale

The broader support-width validation has matching local and Colab artifact-backed evidence. In both backends, top-k 2 support improves ordinary alpha-0 supervised CE loss and final residual loss over the top-k 1 baseline at larger-char and tokenized scales while leaving the supervised CE objective and temporal-clipped HEP path fixed. This supports continuing support-width validation, not changing the residual objective.

## Evidence

| Backend | Scale | Artifact check | Verdict | Baseline alpha-0 | Support alpha-0 | Delta | Baseline final | Support final | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local | larger_char | `pass` | `pass` | 3.39831114 | 3.11434245 | -0.28396869 | 3.39831114 | 3.11434245 | -0.28396869 |
| local | tokenized | `pass` | `pass` | 4.05925131 | 3.54048610 | -0.51876521 | 4.05925131 | 3.54048610 | -0.51876521 |
| colab | larger_char | `pass` | `pass` | 3.39831114 | 3.15306783 | -0.24524331 | 3.39831114 | 3.15306783 | -0.24524331 |
| colab | tokenized | `pass` | `pass` | 4.05925083 | 3.52541637 | -0.53383446 | 4.05925083 | 3.52541637 | -0.53383446 |

## Next Step

define a bounded repeat or capacity-interaction support-width validation gate before any default support-width change
