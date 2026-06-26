# Promoted Top-k-2 Retention Synthesis Gate

- Status: `pass`
- Decision: `retention_separability_risk_mitigation_recommended`
- Mean promoted top-k-2 transfer CE improvement: `0.9278541803359985`
- Mean random-fixed top-k-2 transfer CE improvement: `0.6497141122817993`
- Mean dense transfer CE improvement: `0.4819493293762207`
- Mean promoted top-k-2 support churn after transfer: `0.87109375`
- Mean rank-matched top-k-1 support churn after transfer: `0.005859375`
- Minimum top-k-2/top-k-1 commutator logit-MSE ratio: `22.61928319742693`
- Oracle support regret: `0.002528395038098097`

Across the two fetched RunPod task-free retention packets, promoted contextual top-k-2 remains a strong train-time transfer router versus random fixed top-k-2 and dense active-rank controls, while support selection quality is already established by low oracle-support regret. The same packets replicate high support churn and a much larger finite-update commutator than rank-matched top-k-1, and the deconfounded causal packet does not support broad top-k-2 causal cooperation. Another seed would mostly measure stability of a known risk; the higher-information next step is a bounded support-stability or finite-update mitigation experiment.

Next step: run a bounded support-stability or finite-update mitigation probe against promoted contextual top-k-2, with rank-matched top-k-1, random fixed top-k-2, and dense active-rank controls retained
