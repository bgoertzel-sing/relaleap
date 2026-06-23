# Residual Capacity Support Colab Decision

- Status: `pass`
- Decision: `continue_residual_capacity_support_validation`
- Selected direction: `residual_capacity_support_validation_gate`
- Default residual objective: `supervised_ce`

## Rationale

The residual capacity/support diagnostic now has matching local and Colab artifact-backed evidence. In both backends, widened support is the best variant and accepts a nonzero temporal-clipped HEP alpha inside the ordinary-logit budget, while increased column capacity alone does not explain the gain. This supports continuing with a broader support-width validation gate, not changing the default residual objective yet.

## Evidence

| Backend | Artifact check | Verdict | Best variant | Support minus baseline | Accepted support alpha |
| --- | --- | --- | --- | ---: | ---: |
| local | `pass` | `pass` | `support_width` | -0.08517456 | 1.00000000 |
| colab | `pass` | `pass` | `support_width` | -0.09282660 | 1.00000000 |

## Next Step

define a command-driven support-width validation gate at larger char and tokenized scales
