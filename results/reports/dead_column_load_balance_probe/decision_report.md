# Dead-Column Load-Balance Probe

- Status: `pass`
- Decision: `keep_router_load_balance_probe_opt_in`
- Promote router load-balance default: `False`
- Successful probes: `2` / `2`

## Rationale

Both local tokenized promoted-default dead-column bracket probes recruit additional columns within the configured CE tolerance, so router load balancing is a plausible train-time utilization probe. It remains opt-in because the matched causal fingerprint audits do not show a cleaner reusable-column interpretation: mean absolute ablation effects are essentially unchanged, direct force interventions remain strongly disruptive, and fixed support-pair swaps remain much worse than the learned per-token router.

## Evidence

| Probe | Seed | Baseline CE | Selected | Weight | Selected CE | Used columns | Unique supports |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `token_larger_support_wide_hep_temporal_clipped_objective_gate_dead_column_probe` | 1 | 2.91240144 | `load_balance_0.0125` | 0.01250000 | 2.83027077 | 24 | 45 |
| `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2_dead_column_probe` | 2 | 2.89844608 | `load_balance_0.02` | 0.02000000 | 2.89693952 | 24 | 48 |

## Causal Fingerprints

| Audit | Seed | Selected | Used columns | Mean abs ablate delta | Mean abs force delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `token_larger_support_wide_hep_temporal_clipped_objective_gate_causal_column_fingerprint` | 1 | `load_balance_0.0125` | 24 | 0.05555601 | 1.41382128 |
| `token_larger_support_wide_hep_temporal_clipped_objective_gate_seed2_causal_column_fingerprint` | 2 | `load_balance_0.02` | 24 | 0.05554788 | 1.41234569 |

## Next Step

stop load-balancing promotion work for now and run the promoted-contextual-router support deconfounding controls: rank-matched top-k-1, learned top-k-2, random fixed top-k-2, dense rank/FLOP-matched residuals, and residual-sum normalization variants with oracle-regret, functional-churn, residual-norm, support-margin, and causal-fingerprint outputs
