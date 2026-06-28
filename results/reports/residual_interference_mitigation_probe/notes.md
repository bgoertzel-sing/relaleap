# Residual-Interference Mitigation Probe

- Status: `pass`
- Decision: `residual_interference_mitigation_probe_recorded`
- Claim status: `support_width_mitigation_partial_candidate_not_promoted`
- Mitigation under test: `contextual_topk2_support_width_expansion`
- Source summary: `results/reports/mechanism_factorized_continual_learning_probe/summary.json`
- Top-k2 minus top-k1 target CE delta: `-0.0756457805633545`
- Top-k2 minus dense off-target CE drift: `-0.09026687145233155`
- Top-k2 minus dense off-target KL: `-1.473929725587368`
- Top-k2 minus random-support-null target CE delta: `-0.08091492652893068`

This report treats support-width expansion as a residual-interference mitigation screen. It consumes the command-generated known-rule mechanism CL artifact and keeps dense and random-support controls in the gate.

## Next Step

design one sparse target-adaptation rescue that closes the remaining dense gap without losing top-k2 off-target advantages
