# Residual Support Width Promotion Gate Satisfaction

- Status: `pass`
- Decision: `satisfy_residual_support_width_promotion_gate`
- Promotion gate satisfied: `True`
- Promote support-width default: `True`
- Selected support width top-k: `2`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`
- Promotion gate report: `results/reports/residual_support_width_promotion_gate/decision_report.json`
- Promotion gate status: `pass`
- Promotion gate decision: `define_residual_support_width_promotion_gate`

## Rationale

The seed-3 support-width promotion gate has matching local and Colab artifact-backed evidence. In both backends, top-k 2 support improves ordinary alpha-0 supervised CE loss and final residual loss over top-k 1 at larger-char and tokenized scales while holding supervised CE and temporal-clipped HEP fixed.

## Evidence

| Backend | Scale | Artifact check | Verdict | Baseline alpha-0 | Support alpha-0 | Delta | Baseline final | Support final | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local | larger_char | `pass` | `pass` | 3.50802588 | 3.10037756 | -0.40764832 | 3.50802588 | 3.10037756 | -0.40764832 |
| local | tokenized | `pass` | `pass` | 4.17101955 | 3.59574461 | -0.57527494 | 4.17101955 | 3.59574461 | -0.57527494 |
| colab | larger_char | `pass` | `pass` | 3.50802565 | 3.20798969 | -0.30003595 | 3.50802565 | 3.20798969 | -0.30003596 |
| colab | tokenized | `pass` | `pass` | 4.17101955 | 3.63502359 | -0.53599596 | 4.17101955 | 3.63502359 | -0.53599596 |

## Next Step

change the default residual support width to top-k 2 in the focused support-width configs and run the relevant local artifact checks
