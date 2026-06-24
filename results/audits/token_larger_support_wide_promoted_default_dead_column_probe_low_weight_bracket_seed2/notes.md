# Dead-Column Recruitment Probe

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2_dead_column_probe`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2.yaml`
- Status: `ok`
- Decision: `recruited_without_ce_hurt`
- Baseline alpha-0 CE loss: `2.8984460830688477`
- Baseline used columns: `23`
- CE tolerance: `0.01`
- Selected variant: `load_balance_0.02`

## Variants

- `baseline`: alpha-0 CE `2.8984460830688477`, used columns `23`, dead columns `1`, unique support sets `56`, effective columns `16.390146304864324`, support churn `0.0`, random-support CE `4.222525119781494`, dense-uniform CE `4.221386909484863`, oracle-support regret `0.009251447394490242`, best fixed support `18,22`
- `load_balance_0.01125`: alpha-0 CE `2.953988790512085`, used columns `24`, dead columns `0`, unique support sets `42`, effective columns `9.033218470020675`, support churn `0.98828125`, random-support CE `4.227893352508545`, dense-uniform CE `4.220707416534424`, oracle-support regret `0.01084993313997984`, best fixed support `4,13`
- `load_balance_0.0125`: alpha-0 CE `2.899409294128418`, used columns `24`, dead columns `0`, unique support sets `50`, effective columns `14.019895175954648`, support churn `0.95703125`, random-support CE `4.237977504730225`, dense-uniform CE `4.220510005950928`, oracle-support regret `3.202567086191266e-06`, best fixed support `4,13`
- `load_balance_0.01375`: alpha-0 CE `2.917938470840454`, used columns `24`, dead columns `0`, unique support sets `60`, effective columns `16.578800910700732`, support churn `0.9765625`, random-support CE `4.24332332611084`, dense-uniform CE `4.219481945037842`, oracle-support regret `0.00018830904446076602`, best fixed support `3,4`
- `load_balance_0.015`: alpha-0 CE `2.9279637336730957`, used columns `24`, dead columns `0`, unique support sets `42`, effective columns `12.466425718090166`, support churn `0.97265625`, random-support CE `4.215451240539551`, dense-uniform CE `4.219509124755859`, oracle-support regret `5.1804004215227906e-06`, best fixed support `3,4`
- `load_balance_0.02`: alpha-0 CE `2.896939516067505`, used columns `24`, dead columns `0`, unique support sets `48`, effective columns `13.82324404134149`, support churn `0.9921875`, random-support CE `4.236458778381348`, dense-uniform CE `4.219194412231445`, oracle-support regret `0.0007046725950203836`, best fixed support `0,4`
