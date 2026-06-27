# token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_sequence_kfold_ablation

K-fold sequence-heldout causal-feature ablation for the promoted contextual router.

- Config: `configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml`
- Decision: `causal_contextual_router_sequence_holdout_candidate`
- Claim status: `causal_feature_safe_router_local_sequence_holdout_supported`
- Folds: `4`
- Future-context loss delta: `0.1743077039718628`
- Promoted-vs-linear loss delta: `-0.6465423107147217`
- Causal-contextual-vs-linear loss delta: `-0.6118446588516235`
- Causal-contextual-vs-promoted-full loss delta: `0.034697651863098145`

## Key Fold Comparisons
- causal_contextual_vs_linear: mean delta `-0.6118446588516235`, left wins `4/4`
- causal_contextual_vs_full_context_oracle_baseline: mean delta `0.034697651863098145`, left wins `0/4`
- full_context_oracle_baseline_vs_linear: mean delta `-0.6465423107147217`, left wins `4/4`

## Mean Heldout Loss
- promoted_contextual_topk2:actual_full_context: `2.8932899832725525` (oracle gap `0.004269897937774658`, causal `False`)
- causal_contextual_topk2:actual_causal_context: `2.9279876351356506` (oracle gap `0.07135218381881714`, causal `True`)
- causal_contextual_topk2:causal_current_past_position: `2.9279876351356506` (oracle gap `0.07135218381881714`, causal `True`)
- causal_contextual_topk2:current_past_no_position: `2.928188443183899` (oracle gap `0.07155299186706543`, causal `True`)
- causal_contextual_topk2:past_context_only: `2.968213200569153` (oracle gap `0.11157774925231934`, causal `True`)
- promoted_contextual_topk2:causal_current_past_position: `3.0675976872444153` (oracle gap `0.17857760190963745`, causal `True`)
- causal_contextual_topk2:current_hidden_only: `3.087203323841095` (oracle gap `0.23056787252426147`, causal `True`)
- promoted_contextual_topk2:past_context_only: `3.118002772331238` (oracle gap `0.22898268699645996`, causal `True`)
- contextual_topk1_control:actual_full_context: `3.1447307467460632` (oracle gap `0.16186970472335815`, causal `False`)
- promoted_contextual_topk2:current_hidden_only: `3.1763316988945007` (oracle gap `0.2873116135597229`, causal `True`)
- causal_contextual_topk1_control:current_past_no_position: `3.188281536102295` (oracle gap `0.1540939211845398`, causal `True`)
- causal_contextual_topk1_control:actual_causal_context: `3.2117589116096497` (oracle gap `0.17757129669189453`, causal `True`)
- causal_contextual_topk1_control:causal_current_past_position: `3.2117589116096497` (oracle gap `0.17757129669189453`, causal `True`)
- linear_topk2_control:linear_actual: `3.539832293987274` (oracle gap `0.017778515815734863`, causal `True`)
- causal_contextual_topk1_control:past_context_only: `3.742157816886902` (oracle gap `0.7079702019691467`, causal `True`)
- contextual_topk1_control:causal_current_past_position: `4.018101453781128` (oracle gap `1.0352404117584229`, causal `True`)
- causal_contextual_topk1_control:current_hidden_only: `4.0509703159332275` (oracle gap `1.0167827010154724`, causal `True`)
- contextual_topk1_control:past_context_only: `4.077122688293457` (oracle gap `1.094261646270752`, causal `True`)
- contextual_topk1_control:current_hidden_only: `4.149282813072205` (oracle gap `1.1664217710494995`, causal `True`)
- causal_contextual_topk2:position_only: `4.164309144020081` (oracle gap `1.307673692703247`, causal `True`)
- promoted_contextual_topk2:position_only: `4.219200849533081` (oracle gap `1.3301807641983032`, causal `True`)
- contextual_topk1_control:position_only: `4.243008613586426` (oracle gap `1.2601475715637207`, causal `True`)
- causal_contextual_topk1_control:position_only: `4.260223031044006` (oracle gap `1.2260354161262512`, causal `True`)
