# Dead-Column Recruitment Probe

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_dead_column_probe`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml`
- Status: `ok`
- Decision: `recruited_without_ce_hurt`
- Baseline alpha-0 CE loss: `2.9124014377593994`
- Baseline used columns: `19`
- CE tolerance: `0.01`
- Selected variant: `load_balance_0.0125`

## Variants

- `baseline`: alpha-0 CE `2.9124014377593994`, used columns `19`, dead columns `5`, unique support sets `41`
- `load_balance_0.0025`: alpha-0 CE `3.0304391384124756`, used columns `24`, dead columns `0`, unique support sets `47`
- `load_balance_0.005`: alpha-0 CE `2.9749927520751953`, used columns `24`, dead columns `0`, unique support sets `46`
- `load_balance_0.0075`: alpha-0 CE `2.9234910011291504`, used columns `24`, dead columns `0`, unique support sets `55`
- `load_balance_0.01`: alpha-0 CE `2.929391384124756`, used columns `24`, dead columns `0`, unique support sets `53`
- `load_balance_0.0125`: alpha-0 CE `2.830270767211914`, used columns `24`, dead columns `0`, unique support sets `45`
