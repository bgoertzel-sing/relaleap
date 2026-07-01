# Transformer-ACSR Hidden/Future Support-Value Headroom

- Decision: `support_value_headroom_negligible_close_teacher_imitation_before_gpu`
- Claim status: `teacher_support_imitation_has_insufficient_same_student_value_headroom`
- Train mean oracle-router gap: `5.2226103352310974e-05`
- Heldout mean oracle-router gap: `0.00015570057762993706`
- Value-target training allowed: `False`
- Next step: `close_transformer_acsr_teacher_support_imitation_and_select_next_local_mechanism`

This audit uses exact same-student forced-support losses only. It does
not train a router and does not permit GPU validation by itself.
