# Promoted Top-k-2 Causal Adequacy Matrix

Status: `pass`
Decision: `predictive_default_causal_adequacy_not_established`

This is a no-training synthesis over command-generated artifacts. It compares promoted contextual top-k-2 against rank-matched contextual top-k-1, random fixed top-k-2, and dense active-rank controls.

- Top-k-2 transfer advantage vs random fixed top-k-2: `0.27898430824279785`
- Top-k-2 transfer advantage vs dense active-rank: `0.4466829299926758`
- Top-k-2 CE deficit vs top-k-1: `0.045946598052978516`
- Top-k-2 minus top-k-1 support churn: `0.857421875`
- Top-k-2/top-k-1 finite-update logit-MSE ratio: `25.820599091934366`
- Oracle support regret: `0.002528395038098097`

Promoted contextual top-k-2 remains the predictive support-routing default because it beats random fixed top-k-2 and dense active-rank controls on transfer and has small oracle-support regret. It does not pass the stronger causal-adequacy gate: support churn and finite-update logit/residual risk are high versus rank-matched top-k-1, and deconfounded intervention strata do not clear the pre-registered causal-cooperation threshold. Rank-matched top-k-1 therefore stays a retention/churn control, not the default router.

Next step: run the already-selected local no-training finite-update order-symmetrization audit for promoted contextual top-k-2
