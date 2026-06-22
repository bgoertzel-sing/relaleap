# Focal Residual Objective Promotion Gate

- Status: `pass`
- Decision: `define_focal_residual_objective_promotion_gate`
- Promote residual learning method: `False`
- Selected variant: `supervised_ce_focal`
- Default residual objective: `supervised_ce`
- Focal decision report: `results/reports/focal_residual_objective_decision/decision_report.json`
- Focal decision status: `pass`
- Focal decision: `continue_focal_residual_objective_validation`
- Mean focal minus supervised best HEP loss: `-0.00066582`
- Mean focal minus supervised final residual loss: `-0.22296045`

## Rationale

Focal CE has beaten supervised CE on best temporal-clipped HEP supervised CE loss across the current artifact-backed local and Colab validation, extended, larger, tokenized larger, xlarge, and xxlarge gates. The next gate should test seed robustness at the largest char scale and at the tokenized scale before any default residual objective change.

## Required Evidence

| Gate | Dataset | Backends | Minimum seq len | Minimum hidden dim | Minimum columns | Minimum steps |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| char_xxlarge_seed2_local_colab | tiny_shakespeare_char | local, colab | 192 | 160 | 40 | 70 |
| token_larger_seed2_local_colab | tiny_shakespeare_word | local, colab | 64 | 96 | 24 | 50 |

## Next Step

add seed-2 focal objective-gate configs for xxlarge char and token larger settings
