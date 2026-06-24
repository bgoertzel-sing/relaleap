# Dead-Column Recruitment Probe

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2_dead_column_probe`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml`
- Status: `ok`
- Decision: `recruited_without_ce_hurt`
- Baseline alpha-0 CE loss: `2.8984463214874268`
- Baseline used columns: `23`
- CE tolerance: `0.01`
- Selected variant: `load_balance_0.02`

## Variants

- `baseline`: alpha-0 CE `2.8984463214874268`, used columns `23`, dead columns `1`, unique support sets `56`, effective columns `16.390146304864324`, support churn `0.0`, random-support CE `4.222519397735596`, dense-uniform CE `4.221386909484863`
- `load_balance_0.01125`: alpha-0 CE `2.9618136882781982`, used columns `24`, dead columns `0`, unique support sets `47`, effective columns `12.881769041769042`, support churn `1.0`, random-support CE `4.229682445526123`, dense-uniform CE `4.220709323883057`
- `load_balance_0.0125`: alpha-0 CE `2.899409055709839`, used columns `24`, dead columns `0`, unique support sets `50`, effective columns `14.019895175954648`, support churn `0.95703125`, random-support CE `4.237977027893066`, dense-uniform CE `4.220510482788086`
- `load_balance_0.01375`: alpha-0 CE `2.917938470840454`, used columns `24`, dead columns `0`, unique support sets `60`, effective columns `16.578800910700732`, support churn `0.9765625`, random-support CE `4.24332332611084`, dense-uniform CE `4.219481468200684`
- `load_balance_0.015`: alpha-0 CE `2.927964448928833`, used columns `24`, dead columns `0`, unique support sets `42`, effective columns `12.466425718090166`, support churn `0.97265625`, random-support CE `4.215456008911133`, dense-uniform CE `4.219509124755859`
- `load_balance_0.02`: alpha-0 CE `2.896939516067505`, used columns `24`, dead columns `0`, unique support sets `48`, effective columns `13.82324404134149`, support churn `0.9921875`, random-support CE `4.2364583015441895`, dense-uniform CE `4.219194412231445`
