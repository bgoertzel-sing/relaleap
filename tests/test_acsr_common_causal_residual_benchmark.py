from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_common_causal_residual_benchmark import (
    REQUIRED_ARTIFACTS,
    _benchmark_gate_rows,
    run_acsr_common_causal_residual_benchmark,
)


class ACSRCommonCausalResidualBenchmarkTest(unittest.TestCase):
    def test_missing_source_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_common_causal_residual_benchmark(
                source_probe_dir=root / "missing",
                config_path=root / "missing.yaml",
                out_dir=root / "out",
                train_steps=1,
                dense_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_common_causal_residual_benchmark_failed_closed")
            self.assertEqual(summary["claim_status"], "benchmark_not_run")
            self.assertTrue(any(row["criterion"] == "source_probe_present" for row in summary["failures"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_gate_requires_common_baselines_and_sparse_dense_separation(self) -> None:
        arm_rows = [
            _arm("base_no_residual", 0.0, 0.0),
            _arm("sparse_contextual_topk2", -0.12, 0.4, heldout_ce=4.0),
            _arm("sparse_rank_matched_topk1", -0.04, 0.3),
            _arm("rank_flop_matched_causal_dense", -0.05, 0.45),
            _arm("rank_flop_matched_token_position_dense", 0.01, 0.25),
            _arm("sparse_frequency_matched_random", -0.01, 0.4, heldout_ce=4.2),
            _arm("sparse_shuffled_support_marginals", -0.02, 0.4),
            _arm("sparse_token_position_null", 0.03, 0.4),
            _arm("sparse_oracle_support", -0.2, 0.4, heldout_ce=3.9),
        ]
        gate_rows = _benchmark_gate_rows(arm_rows, [{"fingerprint": str(i)} for i in range(5)])

        self.assertTrue(all(row["passed"] for row in gate_rows), gate_rows)

    def test_gate_fails_when_dense_matches_sparse(self) -> None:
        arm_rows = [
            _arm("base_no_residual", 0.0, 0.0),
            _arm("sparse_contextual_topk2", -0.03, 0.4, heldout_ce=4.0),
            _arm("sparse_rank_matched_topk1", -0.02, 0.3),
            _arm("rank_flop_matched_causal_dense", -0.08, 0.45),
            _arm("rank_flop_matched_token_position_dense", 0.01, 0.25),
            _arm("sparse_frequency_matched_random", 0.0, 0.4, heldout_ce=4.1),
            _arm("sparse_shuffled_support_marginals", -0.01, 0.4),
            _arm("sparse_token_position_null", 0.02, 0.4),
            _arm("sparse_oracle_support", -0.1, 0.4, heldout_ce=3.9),
        ]
        gate_rows = _benchmark_gate_rows(arm_rows, [{"fingerprint": str(i)} for i in range(5)])

        self.assertTrue(
            any(row["criterion"] == "sparse_beats_causal_dense" and not row["passed"] for row in gate_rows)
        )


def _arm(name: str, delta: float, l2: float, *, heldout_ce: float = 4.0) -> dict[str, object]:
    return {
        "arm": name,
        "heldout_delta_vs_base_ce": delta,
        "heldout_residual_update_l2": l2,
        "heldout_ce_loss": heldout_ce,
    }


if __name__ == "__main__":
    unittest.main()
