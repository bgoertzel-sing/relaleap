# Promoted Top-k-2 Retention Reference Audit

- Status: `pass`
- Decision: `promoted_topk2_router_default_retention_reference`
- Mean top-k-2 support churn: `0.85546875`
- Mean top-k-2 transfer CE improvement: `0.9277406930923462`
- Minimum top-k-2 transfer improvement over dense: `0.48139703273773193`
- Mean top-k-2 support churn minus top-k-1: `0.849609375`
- Top-k-1 context gate failed: `True`

## Rationale

The context-gate suppression audit failed, so top-k-1 singleton gating stays diagnostic-only. The existing retention/churn packets still support keeping promoted contextual top-k-2 as the main router default for CE and support-selection evidence, but not as a low-churn causal-retention mechanism: top-k-2 improves transfer CE over the dense control while showing high support churn and larger finite-update commutator drift than the rank-matched top-k-1 bracket.

## Next Step

run one local no-training audit on the promoted contextual top-k-2 router that targets support-selection quality rather than reusable singleton gating or low-churn retention
