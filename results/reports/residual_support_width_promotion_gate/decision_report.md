# Residual Support Width Promotion Gate

- Status: `pass`
- Decision: `define_residual_support_width_promotion_gate`
- Selected direction: `support_width_seed3_promotion_gate_evidence`
- Promote support-width default: `False`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`
- Repeat decision report: `results/reports/residual_support_width_repeat_decision/decision_report.json`
- Repeat decision status: `pass`
- Repeat decision: `satisfy_residual_support_width_repeat_gate`

## Rationale

Support width top-k 2 has passed the first larger-char/tokenized validation and the seed-2 repeat in both local and Colab artifact trees. The default support width should still not change until a bounded seed-3 promotion gate confirms the same ordinary alpha-0 and final residual CE improvements at both scales and backends, with supervised CE and temporal-clipped HEP held fixed.

## Required Evidence

| Gate | Dataset | Backends | Minimum seq len | Minimum hidden dim | Minimum columns | Minimum steps | Baseline top-k | Support top-k |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| larger_char_seed3_local_colab | tiny_shakespeare_char | local, colab | 128 | 96 | 24 | 50 | 1 | 2 |
| tokenized_seed3_local_colab | tiny_shakespeare_word | local, colab | 64 | 96 | 24 | 50 | 1 | 2 |

## Next Step

add seed-3 support-width promotion-gate configs for larger-char and tokenized local/Colab evidence
