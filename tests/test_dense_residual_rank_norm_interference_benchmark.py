from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_residual_rank_norm_interference_benchmark import (
    REQUIRED_ARTIFACTS,
    run_dense_residual_rank_norm_interference_benchmark,
)


class DenseResidualRankNormInterferenceBenchmarkTest(unittest.TestCase):
    def test_missing_common_artifacts_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_residual_rank_norm_interference_benchmark(
                common_benchmark_dir=root / "missing",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "dense_residual_rank_norm_interference_failed_closed")
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_passes_when_dense_wins_without_extra_damage_at_matched_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            common.mkdir()
            (common / "summary.json").write_text(json.dumps({"status": "fail"}) + "\n", encoding="utf-8")
            _write_csv(
                common / "arm_metrics.csv",
                [
                    _arm("sparse_contextual_topk2", "sparse", -0.30, 1.0, active=200, flops=200, top_k=2),
                    _arm("sparse_rank_matched_topk1", "sparse", -0.20, 1.0, active=100, flops=100, top_k=1),
                    _arm("sparse_frequency_matched_random_topk1", "sparse", -0.05, 1.0, active=100, flops=100, top_k=1),
                    _arm("rank_flop_matched_causal_dense", "dense", -0.40, 1.0, active=9000, flops=9000, rank=24),
                    _arm(
                        "rank_flop_matched_token_position_dense",
                        "dense",
                        0.01,
                        1.0,
                        active=9000,
                        flops=9000,
                        rank=94,
                    ),
                    _arm(
                        "rank_flop_matched_shuffled_causal_feature_dense_null",
                        "dense",
                        0.02,
                        1.0,
                        active=9000,
                        flops=9000,
                        rank=94,
                    ),
                    _arm(
                        "rank_flop_matched_ablated_context_dense",
                        "dense",
                        0.03,
                        1.0,
                        active=9000,
                        flops=9000,
                        rank=94,
                    ),
                ],
            )
            _write_csv(
                common / "per_token_metrics.csv",
                _per_token("sparse_contextual_topk2", [-0.2, -0.4], [-0.2, -0.4])
                + _per_token("sparse_rank_matched_topk1", [-0.1, -0.3], [-0.1, -0.3])
                + _per_token("sparse_frequency_matched_random_topk1", [0.0, -0.1], [0.0, -0.1])
                + _per_token("rank_flop_matched_causal_dense", [-0.3, -0.5], [-0.3, -0.5])
                + _per_token("rank_flop_matched_token_position_dense", [0.1, -0.1], [0.1, -0.1])
                + _per_token("rank_flop_matched_shuffled_causal_feature_dense_null", [0.1, 0.0], [0.1, 0.0])
                + _per_token("rank_flop_matched_ablated_context_dense", [0.2, 0.0], [0.2, 0.0]),
            )

            summary = run_dense_residual_rank_norm_interference_benchmark(
                common_benchmark_dir=common,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], "causal_dense_control_remains_active_local_baseline")
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            self.assertTrue(
                any(
                    row["row_type"] == "paired_arms"
                    and row["arm"] == "rank_flop_matched_causal_dense"
                    and row["reference_arm"] == "sparse_contextual_topk2"
                    for row in summary["interference_rows"]
                )
            )

    def test_fails_when_dense_norm_is_not_matched(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            common.mkdir()
            (common / "summary.json").write_text(json.dumps({"status": "fail"}) + "\n", encoding="utf-8")
            _write_csv(
                common / "arm_metrics.csv",
                [
                    _arm("sparse_contextual_topk2", "sparse", -0.30, 1.0, active=200, flops=200, top_k=2),
                    _arm("sparse_rank_matched_topk1", "sparse", -0.20, 1.0, active=100, flops=100, top_k=1),
                    _arm("sparse_frequency_matched_random_topk1", "sparse", -0.05, 1.0, active=100, flops=100, top_k=1),
                    _arm("rank_flop_matched_causal_dense", "dense", -0.40, 1.5, active=9000, flops=9000, rank=24),
                    _arm("rank_flop_matched_token_position_dense", "dense", 0.01, 1.0, active=9000, flops=9000, rank=94),
                    _arm("rank_flop_matched_shuffled_causal_feature_dense_null", "dense", 0.02, 1.0, active=9000, flops=9000, rank=94),
                    _arm("rank_flop_matched_ablated_context_dense", "dense", 0.03, 1.0, active=9000, flops=9000, rank=94),
                ],
            )
            _write_csv(
                common / "per_token_metrics.csv",
                _per_token("sparse_contextual_topk2", [-0.2, -0.4], [-0.2, -0.4])
                + _per_token("sparse_rank_matched_topk1", [-0.1, -0.3], [-0.1, -0.3])
                + _per_token("sparse_frequency_matched_random_topk1", [0.0, -0.1], [0.0, -0.1])
                + _per_token("rank_flop_matched_causal_dense", [-0.3, -0.5], [-0.3, -0.5])
                + _per_token("rank_flop_matched_token_position_dense", [0.1, -0.1], [0.1, -0.1])
                + _per_token("rank_flop_matched_shuffled_causal_feature_dense_null", [0.1, 0.0], [0.1, 0.0])
                + _per_token("rank_flop_matched_ablated_context_dense", [0.2, 0.0], [0.2, 0.0]),
            )

            summary = run_dense_residual_rank_norm_interference_benchmark(
                common_benchmark_dir=common,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(
                    row["criterion"] == "dense_norm_matched_to_sparse_topk2" and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )


def _arm(
    arm: str,
    family: str,
    delta: float,
    l2: float,
    *,
    active: int,
    flops: int,
    top_k: int | str = "",
    rank: int | str = "",
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "top_k": top_k,
        "rank": rank,
        "heldout_delta_vs_base_ce": delta,
        "heldout_residual_update_l2": l2,
        "active_params_proxy": active,
        "flops_proxy": flops,
    }


def _per_token(arm: str, train_deltas: list[float], heldout_deltas: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    token_index = 0
    for split, deltas in (("train", train_deltas), ("heldout", heldout_deltas)):
        for delta in deltas:
            rows.append(
                {
                    "arm": arm,
                    "token_index": token_index,
                    "position_index": token_index,
                    "split": split,
                    "base_ce_loss": 1.0,
                    "ce_loss": 1.0 + delta,
                    "delta_vs_base_ce": delta,
                    "residual_update_l2": 1.0,
                }
            )
            token_index += 1
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
