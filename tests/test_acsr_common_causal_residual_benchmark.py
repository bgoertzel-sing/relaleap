from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_common_causal_residual_benchmark import (
    REQUIRED_ARTIFACTS,
    _benchmark_gate_rows,
    _norm_sweep_rows,
    _summary,
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

    def test_seed2_compute_mismatch_selects_local_repair_not_runpod(self) -> None:
        arm_rows = [
            _arm("base_no_residual", 0.0, 0.0, active_params=0),
            _arm("sparse_contextual_topk2", -0.3168487548828125, 1.0178145170211792, active_params=192),
            _arm("sparse_rank_matched_topk1", -0.2, 0.8, active_params=96),
            _arm("rank_flop_matched_causal_dense", -0.4195396900177002, 1.0178145170211792, active_params=9288),
            _arm("rank_flop_matched_token_position_dense", 0.029190540313720703, 1.0178143978118896, active_params=9306),
            _arm("sparse_frequency_matched_random", 0.0, 1.0, active_params=192, heldout_ce=4.2),
            _arm("sparse_shuffled_support_marginals", -0.01, 1.0, active_params=192),
            _arm("sparse_token_position_null", 0.02, 1.0, active_params=192),
            _arm("sparse_oracle_support", -0.3192157745361328, 1.0178145170211792, active_params=192, heldout_ce=3.9),
        ]
        gate_rows = _benchmark_gate_rows(arm_rows, [{"fingerprint": str(i)} for i in range(5)])
        compute_gate = next(row for row in gate_rows if row["criterion"] == "active_compute_matched_or_bracketed")

        self.assertFalse(compute_gate["passed"])
        self.assertEqual(compute_gate["actual"]["sparse_active_params_proxy"], 192)
        self.assertEqual(compute_gate["actual"]["dense_active_params_proxy"], [9288, 9306])
        self.assertGreater(compute_gate["actual"]["dense_to_sparse_active_ratios"][0], 48.0)

        summary = _summary(
            status="fail",
            decision="acsr_common_causal_residual_benchmark_failed_gate",
            claim_status="sparse_support_specific_effect_not_separated_from_common_dense_controls",
            start=0.0,
            source_probe_dir=Path("source"),
            config_path=Path("config.yaml"),
            train_steps=12,
            dense_steps=80,
            arm_rows=arm_rows,
            per_token_rows=[],
            norm_rows=[],
            fingerprint_rows=[],
            gate_rows=gate_rows,
            out_dir=Path("out"),
        )
        self.assertIn("repair local compute-matched", summary["selected_next_step"])
        self.assertNotIn("RunPod repeat", summary["selected_next_step"])

    def test_dense_bottleneck_ladder_brackets_sparse_active_compute(self) -> None:
        arm_rows = [
            _arm("base_no_residual", 0.0, 0.0, active_params=0),
            _arm("sparse_contextual_topk2", -0.12, 0.4, active_params=192),
            _arm("sparse_rank_matched_topk1", -0.04, 0.3, active_params=96),
            _arm("rank_flop_matched_causal_dense", -0.08, 0.4, active_params=9288),
            _arm("dense_bottleneck_causal_rank0", 0.0, 0.0, active_params=0),
            _arm("dense_bottleneck_causal_rank1", -0.03, 0.2, active_params=384),
            _arm("rank_flop_matched_token_position_dense", 0.01, 0.25, active_params=9306),
            _arm("sparse_frequency_matched_random", -0.01, 0.4, active_params=192, heldout_ce=4.2),
            _arm("sparse_shuffled_support_marginals", -0.02, 0.4, active_params=192),
            _arm("sparse_token_position_null", 0.03, 0.4, active_params=192),
            _arm("sparse_oracle_support", -0.2, 0.4, active_params=192, heldout_ce=3.9),
        ]

        gate_rows = _benchmark_gate_rows(arm_rows, [{"fingerprint": str(i)} for i in range(5)])
        compute_gate = next(row for row in gate_rows if row["criterion"] == "active_compute_matched_or_bracketed")
        norm_rows = _norm_sweep_rows(arm_rows)
        rank1_norm = next(row for row in norm_rows if row["arm"] == "dense_bottleneck_causal_rank1")

        self.assertTrue(compute_gate["passed"])
        self.assertTrue(compute_gate["actual"]["bracketed_by_dense_ladder"])
        self.assertEqual(compute_gate["actual"]["matched_dense_count"], 1)
        self.assertAlmostEqual(rank1_norm["active_ratio_vs_sparse_topk2"], 2.0)
        self.assertIn("active_compute_pareto_front", rank1_norm)


def _arm(
    name: str,
    delta: float,
    l2: float,
    *,
    heldout_ce: float = 4.0,
    active_params: int | None = None,
) -> dict[str, object]:
    if active_params is None:
        active_params = 192 if "dense" not in name else 200
    return {
        "arm": name,
        "family": "dense" if "dense" in name else ("base" if name == "base_no_residual" else "sparse"),
        "heldout_delta_vs_base_ce": delta,
        "heldout_residual_update_l2": l2,
        "heldout_ce_loss": heldout_ce,
        "active_params_proxy": active_params,
        "flops_proxy": active_params,
    }


if __name__ == "__main__":
    unittest.main()
