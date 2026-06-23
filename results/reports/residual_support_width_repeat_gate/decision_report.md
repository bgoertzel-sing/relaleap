# Residual Support Width Repeat Gate

- Status: `pass`
- Decision: `define_residual_support_width_repeat_gate`
- Selected direction: `support_width_larger_char_token_seed2_repeat`
- Default residual objective: `supervised_ce`
- Default support-stress mitigation: `temporal_clipped_hep`
- Config count: `4`

## Rationale

The first larger-char/tokenized support-width validation selected continued validation, not a default change. This gate bounds the next step to a seed-2 repeat of the same top-k 1 versus top-k 2 matrix under supervised CE and promoted temporal-clipped HEP. A default support-width change remains blocked until matching local and Colab repeat evidence exists.

## Commands

```bash
python -m relaleap.experiments.compare --config configs/char_larger_hep_temporal_clipped_objective_gate_seed2.yaml --config configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml --config configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml --config configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml --out results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2
python -m relaleap.experiments.check_artifacts --comparison-dir results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2 --out results/comparisons/support_width_larger_char_token_temporal_clipped_objective_gate_seed2/artifact_check_local.json
```

## Config Matrix

| Scale | Support-wide | Seed | Config | Dataset | Seq len | Columns | Top-k | Failures |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| larger_char | `False` | 2 | `configs/char_larger_hep_temporal_clipped_objective_gate_seed2.yaml` | tiny_shakespeare_char | 128 | 24 | 1 | 0 |
| larger_char | `True` | 2 | `configs/char_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml` | tiny_shakespeare_char | 128 | 24 | 2 | 0 |
| tokenized | `False` | 2 | `configs/token_larger_hep_temporal_clipped_objective_gate_seed2.yaml` | tiny_shakespeare_word | 64 | 24 | 1 | 0 |
| tokenized | `True` | 2 | `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml` | tiny_shakespeare_word | 64 | 24 | 2 | 0 |

## Next Step

run the local seed-2 support-width repeat comparison and artifact check recorded in commands.compare and commands.check_artifacts
