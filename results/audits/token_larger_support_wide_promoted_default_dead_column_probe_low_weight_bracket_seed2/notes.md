# Dead-Column Recruitment Probe

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2_dead_column_probe`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml`
- Status: `ok`
- Decision: `recruited_without_ce_hurt`
- Baseline alpha-0 CE loss: `2.8984458446502686`
- Baseline used columns: `23`
- CE tolerance: `0.01`
- Selected variant: `load_balance_0.02`

## Variants

- `baseline`: alpha-0 CE `2.8984458446502686`, used columns `23`, dead columns `1`, unique support sets `56`
- `load_balance_0.01125`: alpha-0 CE `2.9618136882781982`, used columns `24`, dead columns `0`, unique support sets `47`
- `load_balance_0.0125`: alpha-0 CE `2.899409055709839`, used columns `24`, dead columns `0`, unique support sets `50`
- `load_balance_0.01375`: alpha-0 CE `2.917938232421875`, used columns `24`, dead columns `0`, unique support sets `60`
- `load_balance_0.015`: alpha-0 CE `2.927964210510254`, used columns `24`, dead columns `0`, unique support sets `42`
- `load_balance_0.02`: alpha-0 CE `2.896939277648926`, used columns `24`, dead columns `0`, unique support sets `48`
