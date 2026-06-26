# Retention/Churn Microtest

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_retention_churn_microtest`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml`
- Status: `ok`
- Training steps per slice: `50`
- Lowest anchor CE drift: `value_only_transfer_topk2` (-0.9339891672134399)

This local diagnostic trains each control on anchor slice A, continues on transfer slice B, and measures anchor CE/logit/residual drift plus sparse support churn.
