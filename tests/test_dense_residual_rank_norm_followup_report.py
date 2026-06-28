from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_residual_rank_norm_followup_report import (
    REQUIRED_ARTIFACTS,
    run_dense_residual_rank_norm_followup_report,
)


class DenseResidualRankNormFollowupReportTest(unittest.TestCase):
    def test_missing_sources_fail_closed_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_residual_rank_norm_followup_report(
                common_benchmark_dir=root / "missing_common",
                rank_norm_benchmark_dir=root / "missing_rank",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "dense_rank_norm_followup_failed_closed")
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_next_rank_norm_matrix_from_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            rank = root / "rank"
            common.mkdir()
            rank.mkdir()
            (common / "summary.json").write_text(json.dumps({"status": "fail"}) + "\n", encoding="utf-8")
            (rank / "summary.json").write_text(json.dumps({"status": "pass"}) + "\n", encoding="utf-8")
            _write_csv(
                common / "arm_metrics.csv",
                [
                    _arm("sparse_contextual_topk2", "sparse", -0.30, 1.0, active=192),
                    _arm("dense_bottleneck_causal_rank0", "dense", 0.0, 0.0, active=0, rank=0),
                    _arm("dense_bottleneck_causal_rank1", "dense", -0.05, 1.0, active=387, rank=1),
                    _arm(
                        "rank_flop_matched_causal_dense",
                        "dense",
                        -0.40,
                        1.0,
                        active=9288,
                        rank=24,
                        raw_l2=1.2,
                        scale=0.83,
                    ),
                ],
            )
            _write_csv(
                common / "norm_sweep.csv",
                [
                    _norm("sparse_contextual_topk2", "sparse", "", -0.30, 1.0),
                    _norm("dense_bottleneck_causal_rank0", "dense", 0, 0.0, 0.0),
                    _norm("dense_bottleneck_causal_rank1", "dense", 1, -0.05, 1.0),
                    _norm("rank_flop_matched_causal_dense", "dense", 24, -0.40, 1.0),
                ],
            )
            _write_csv(rank / "gate_criteria.csv", [{"criterion": "ok", "passed": True}])

            summary = run_dense_residual_rank_norm_followup_report(
                common_benchmark_dir=common,
                rank_norm_benchmark_dir=rank,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_rank_norm_followup_matrix_selected")
            self.assertTrue(
                any(row["candidate"] == "dense_causal_rank24_norm_scale_1.00" for row in summary["next_matrix_rows"])
            )
            self.assertTrue(
                any(
                    row["criterion"] == "matched_dense_beats_sparse_but_rank1_does_not"
                    and row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            rank_ladder = (root / "out" / "rank_ladder.csv").read_text(encoding="utf-8")
            self.assertIn("rank_flop_matched_causal_dense", rank_ladder)
            next_matrix = (root / "out" / "next_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("dense_causal_rank4_norm_scale_0.75", next_matrix)


def _arm(
    arm: str,
    family: str,
    delta: float,
    l2: float,
    *,
    active: int,
    rank: int | str = "",
    raw_l2: float | str = "",
    scale: float | str = "",
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "heldout_delta_vs_base_ce": delta,
        "heldout_residual_update_l2": l2,
        "active_params_proxy": active,
        "rank": rank,
        "raw_heldout_residual_update_l2": raw_l2,
        "posthoc_residual_norm_scale": scale,
    }


def _norm(
    arm: str,
    family: str,
    rank: int | str,
    delta: float,
    l2: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "rank": rank,
        "heldout_delta_vs_base_ce": delta,
        "heldout_residual_update_l2": l2,
        "active_compute_pareto_front": False,
    }


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
