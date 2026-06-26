# Promoted Top-k-2 Retention Synthesis Gate

- Status: `pass`
- Decision: `contextual_topk2_router_default_topk1_diagnostic`
- Mean promoted top-k-2 transfer CE improvement: `0.9286322593688965`
- Mean random-fixed top-k-2 transfer CE improvement: `0.6496479511260986`
- Mean dense transfer CE improvement: `0.4819493293762207`
- Mean promoted top-k-2 support churn after transfer: `0.86328125`
- Mean rank-matched top-k-1 support churn after transfer: `0.005859375`
- Minimum top-k-2/top-k-1 commutator logit-MSE ratio: `21.15282622701977`
- Oracle support regret: `0.002528395038098097`

The newest fetched RunPod anchor-retention matrix repeats the same tradeoff: rank-matched contextual top-k-1 is cleaner on support churn and finite-update commutators and is transfer-competitive, but the deployable context-gate suppression audit failed. That blocks a scientific shift to top-k-1 as a reusable singleton mechanism. Promoted contextual top-k-2 should remain the router default for CE and support-selection evidence while top-k-1 stays a diagnostic retention bracket. The next non-duplicative step is to probe finite-update order symmetrization rather than adding more low-rank value or top-k-1 singleton-gate variants.

Next step: run a local no-training finite-update order-symmetrization audit for promoted contextual top-k-2, retaining rank-matched top-k-1, random fixed top-k-2, and dense active-rank controls
