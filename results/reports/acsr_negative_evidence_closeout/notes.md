# ACSR Negative Evidence Closeout

- Status: `pass`
- Decision: `acsr_negative_evidence_closeout_branch_selected`
- Selected action: `demote_acsr_to_diagnostic_status`
- Claim status: `acsr_promotion_path_demoted_to_diagnostic_no_default_change`
- Requires GPU now: `False`
- Next step: treat ACSR as a diagnostic probe; make the next experiment target dense/MLP residual controls rather than ACSR/default-router promotion

ACSR beats simple nulls but fails parameter-matched and retention-churn guardrails, while dense-teacher, dense24/rank-norm, MLP, norm-budgeted, commutator, and CL-repeat evidence do not establish a sparse-specific mechanism

This is a local branch selector. It does not promote ACSR or request RunPod/Colab validation.
