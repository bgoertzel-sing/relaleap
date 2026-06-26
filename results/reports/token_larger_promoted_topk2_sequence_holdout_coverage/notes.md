# Promoted Top-k-2 Sequence-Holdout Coverage

- Status: `pass`
- Decision: `sequence_holdout_support_head_generalization_failed`
- Claim status: `deployable_support_head_sequence_generalization_blocked`
- Position holdout present: `True`
- Sequence-level holdout present: `True`
- Strategy review requested sequence holdout: `True`
- Contextual support-head holdout gap recovery: `0.1893932244974993`
- Sequence support-head intervention minus router loss: `0.016779184341430664`
- Sequence recovery fraction numerically fragile: `True`
- Oracle support regret: `0.002528395038098097`
- Causal-adequacy decision: `predictive_default_causal_adequacy_not_established`

The refreshed exhaustive support audit now includes sequence-level holdout coverage, but the train-time-style contextual support head is worse than the learned router on held-out full sequences. The sequence-head absolute intervention-minus-router CE delta is `0.016779184341430664`; because the router-oracle gap is tiny, recovery fractions are numerically fragile. Treat contextual routing as the operational predictive default from prior local/Colab evidence, but block deployable contextual support-selection claims until K-fold sequence-heldout and causal-feature-safe router evidence exists.

Next step: run a local K-fold sequence-heldout causal-feature ablation of the actual promoted contextual router versus the linear/top-k controls
