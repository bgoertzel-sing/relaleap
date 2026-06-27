# ACSR Two-Seed Local Synthesis

Status: pass
Decision: acsr_two_seed_local_synthesis_recorded
Replication gate: runpod_replication_warranted
GPU backend: runpod

## Rationale

Both local token-larger ACSR smoke packets pass the fail-closed leakage, shuffled-feature, token/position-only, and regret gates. ACSR closes the causal-router to full-context teacher CE gap in both packets, beats the null supports through same-student values, and keeps fixed-teacher churn well below shuffled and token/position controls. This warrants backend replication, not promotion.

## Next Step

run RunPod ACSR replication for the two token-larger seed configs, then fetch artifacts and run the same local synthesis/checks

## Command

`./.venv-conda/bin/python tools/runpod_ssh_runner.py bootstrap && ./.venv-conda/bin/python tools/runpod_ssh_runner.py run --command 'python -m relaleap.experiments.anticipatory_contextual_support_routing --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml --out results/audits/runpod_token_larger_anticipatory_contextual_support_routing && python -m relaleap.experiments.anticipatory_contextual_support_routing --config configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate_seed2.yaml --out results/audits/runpod_token_larger_anticipatory_contextual_support_routing_seed2' && ./.venv-conda/bin/python tools/runpod_ssh_runner.py fetch`
