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

- `baseline`: alpha-0 CE `2.9124014377593994`, used columns `19`, dead columns `5`, unique support sets `41`, effective columns `8.326789911695572`, support churn `0.0`, random-support CE `4.192426681518555`, dense-uniform CE `4.152108669281006`, oracle-support regret `0.000963175087235868`, best fixed support `6,9`
- `load_balance_0.01125`: alpha-0 CE `2.834712505340576`, used columns `24`, dead columns `0`, unique support sets `55`, effective columns `15.50780880265026`, support churn `0.97265625`, random-support CE `4.163344383239746`, dense-uniform CE `4.156022548675537`, oracle-support regret `3.635836037574336e-05`, best fixed support `3,7`
- `load_balance_0.0125`: alpha-0 CE `2.830270767211914`, used columns `24`, dead columns `0`, unique support sets `45`, effective columns `13.147958671882837`, support churn `0.97265625`, random-support CE `4.164117336273193`, dense-uniform CE `4.1553168296813965`, oracle-support regret `0.0002397814387222752`, best fixed support `2,7`
- `load_balance_0.01375`: alpha-0 CE `2.8494396209716797`, used columns `24`, dead columns `0`, unique support sets `50`, effective columns `13.56431749974128`, support churn `0.96484375`, random-support CE `4.159765243530273`, dense-uniform CE `4.153985500335693`, oracle-support regret `6.288526492426172e-05`, best fixed support `2,7`
- `load_balance_0.015`: alpha-0 CE `2.8610787391662598`, used columns `24`, dead columns `0`, unique support sets `55`, effective columns `15.09002993322588`, support churn `0.9609375`, random-support CE `4.1442437171936035`, dense-uniform CE `4.153954982757568`, oracle-support regret `0.004308165516704321`, best fixed support `3,7`
- `load_balance_0.02`: alpha-0 CE `2.8683066368103027`, used columns `24`, dead columns `0`, unique support sets `56`, effective columns `17.98463227222832`, support churn `0.96484375`, random-support CE `4.13433313369751`, dense-uniform CE `4.153652191162109`, oracle-support regret `0.0007439657929353416`, best fixed support `7,14`
