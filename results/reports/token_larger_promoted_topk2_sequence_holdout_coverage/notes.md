# Promoted Top-k-2 Sequence-Holdout Coverage

- Status: `pass`
- Decision: `sequence_holdout_extension_required`
- Claim status: `sequence_level_support_prediction_not_yet_tested`
- Position holdout present: `True`
- Sequence-level holdout present: `False`
- Strategy review requested sequence holdout: `True`
- Contextual support-head holdout gap recovery: `0.1893932244974993`
- Oracle support regret: `0.002528395038098097`
- Causal-adequacy decision: `predictive_default_causal_adequacy_not_established`

The promoted top-k-2 support-selection packet uses even flattened token positions for training and odd flattened token positions for holdout. That is useful but does not satisfy the current strategy review's stricter sequence-level held-out evaluation recommendation. The causal-adequacy matrix can stand as the current matched-control claim blocker, but deployable contextual support-selection claims should remain limited until this artifact extension exists.

Next step: extend relaleap.experiments.support_audit with a sequence-level holdout split for contextual support prediction and rerun the promoted token-larger exhaustive support audit
