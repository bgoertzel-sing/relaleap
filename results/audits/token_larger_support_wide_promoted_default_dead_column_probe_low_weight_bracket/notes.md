# Dead-Column Recruitment Probe

- Experiment: `token_larger_support_wide_hep_temporal_clipped_objective_gate_dead_column_probe`
- Config: `configs/token_larger_support_wide_hep_temporal_clipped_objective_gate.yaml`
- Status: `ok`
- Decision: `recruited_without_ce_hurt`
- Baseline alpha-0 CE loss: `2.912400960922241`
- Baseline used columns: `19`
- CE tolerance: `0.01`
- Selected variant: `load_balance_0.0125`

## Variants

- `baseline`: alpha-0 CE `2.912400960922241`, used columns `19`, dead columns `5`, unique support sets `41`, effective columns `8.326789911695572`, support churn `0.0`, random-support CE `4.192426681518555`, dense-uniform CE `4.152108669281006`
- `load_balance_0.01125`: alpha-0 CE `2.834712505340576`, used columns `24`, dead columns `0`, unique support sets `55`, effective columns `15.50780880265026`, support churn `0.97265625`, random-support CE `4.163344383239746`, dense-uniform CE `4.156022548675537`
- `load_balance_0.0125`: alpha-0 CE `2.830270767211914`, used columns `24`, dead columns `0`, unique support sets `45`, effective columns `13.147958671882837`, support churn `0.97265625`, random-support CE `4.164117336273193`, dense-uniform CE `4.155316352844238`
- `load_balance_0.01375`: alpha-0 CE `2.8494391441345215`, used columns `24`, dead columns `0`, unique support sets `50`, effective columns `13.56431749974128`, support churn `0.96484375`, random-support CE `4.159773349761963`, dense-uniform CE `4.153985500335693`
- `load_balance_0.015`: alpha-0 CE `2.861461877822876`, used columns `24`, dead columns `0`, unique support sets `54`, effective columns `14.68593837535014`, support churn `0.9609375`, random-support CE `4.144626140594482`, dense-uniform CE `4.153955936431885`
- `load_balance_0.02`: alpha-0 CE `2.8683066368103027`, used columns `24`, dead columns `0`, unique support sets `56`, effective columns `17.98463227222832`, support churn `0.96484375`, random-support CE `4.134332656860352`, dense-uniform CE `4.153652667999268`
