# Temporal Clipped HEP Promotion Gate Satisfaction

- Status: `pass`
- Decision: `satisfy_temporal_label_free_support_stress_promotion_gate`
- Promotion gate satisfied: `True`
- Promote to default support-stress mitigation: `True`
- Promotion gate report: `results/reports/temporal_clipped_hep_promotion_gate/decision_report.json`
- Promotion gate status: `pass`
- Promotion gate decision: `define_temporal_label_free_support_stress_promotion_gate`
- Report count: `4`
- Accepted temporal report count: `4`
- Mean temporal loss improvement from alpha 0: `0.00008231`
- Max temporal logit delta from ordinary: `0.00174150`
- Max temporal pinned-vs-repicked logit delta: `0.00397229`
- Max support change fraction: `0.65039062`

## Rationale

The defined promotion gate is satisfied: larger-char and non-char tokenized local/Colab reports all pass, select temporal consistency, show nonzero support repicking, and include accepted nonzero temporal alphas inside both stability budgets.

## Evidence

| Gate | Backend | Status | Artifact check | Selected | Alpha | Loss improvement | Logit delta | Pinned-vs-repicked | Support change | Source |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| larger_char_local_colab | local | pass | pass | True | 1.00000000 | 0.00005436 | 0.00062868 | 0.00319886 | 0.65039062 | `results/reports/temporal_clipped_hep_larger_local_decision/decision_report.json` |
| larger_char_local_colab | colab | pass | pass | True | 1.00000000 | 0.00005412 | 0.00062868 | 0.00319910 | 0.65039062 | `results/reports/temporal_clipped_hep_larger_colab_decision/decision_report.json` |
| non_char_tokenized_local_colab | local | pass | pass | True | 1.00000000 | 0.00011063 | 0.00174150 | 0.00397229 | 0.60546875 | `results/reports/temporal_clipped_hep_token_larger_local_decision/decision_report.json` |
| non_char_tokenized_local_colab | colab | pass | pass | True | 1.00000000 | 0.00011015 | 0.00174144 | 0.00397205 | 0.60546875 | `results/reports/temporal_clipped_hep_token_larger_colab_decision/decision_report.json` |

## Next Step

make the explicit default support-stress mitigation change to temporal clipped HEP
