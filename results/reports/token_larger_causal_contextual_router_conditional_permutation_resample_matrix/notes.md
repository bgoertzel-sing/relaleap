# Conditional-Permutation Resample Matrix

Status: `pass`
Decision: `conditional_permutation_assignment_signal_survives_functional_gate_blocks`
Claim status: `teacher_support_assignment_exceeds_conditional_null_but_functional_claim_blocked`
Selected next step: `keep_causal_router_distillation_promotion_frozen`

## Key metrics
- `observed_student_exact_agreement`: `0.8783068783068783`
- `null_mean_student_exact_agreement`: `0.08519861937830692`
- `student_exact_agreement_effect_vs_null_mean`: `0.7931082589285714`
- `student_exact_agreement_empirical_p_upper`: `0.0013003901170351106`
- `observed_oracle_exact_agreement`: `0.05026455026455026`
- `null_mean_oracle_exact_agreement`: `0.03920200892857159`
- `oracle_exact_agreement_effect_vs_null_mean`: `0.01106254133597867`
- `oracle_exact_agreement_empirical_p_upper`: `0.15734720416124837`
- `teacher_forced_gain_all_tokens`: `-0.023253488792944206`
- `token_position_null_forced_gain_all_tokens`: `-0.02417191760565238`
- `teacher_minus_token_position_null_gain_all_tokens`: `0.0009184288127081734`
- `teacher_forced_gain_disagreement_tokens`: `-0.19108301660288934`
- `token_position_null_forced_gain_disagreement_tokens`: `-0.1293056581331336`
- `teacher_minus_token_position_null_gain_disagreement_tokens`: `-0.06177735846975575`

## Rationale
The conditional-permutation matrix estimates support-label assignment ranks from existing per-token artifacts. It does not create new forced-loss evaluations for each sampled support assignment, so the functional mechanism claim remains governed by the same-student teacher-vs-token-position-null gain (0.0009184288127081734).
