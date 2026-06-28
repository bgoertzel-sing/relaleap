from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.norm_budgeted_churn_strata_synthesis import (
    REQUIRED_ARTIFACTS,
    run_norm_budgeted_churn_strata_synthesis,
)


class NormBudgetedChurnStrataSynthesisTest(unittest.TestCase):
    def test_missing_pilot_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_norm_budgeted_churn_strata_synthesis(
                pilot_dir=root / "missing",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "norm_budgeted_churn_strata_synthesis_failed_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_low_budget_challenger_does_not_warrant_runpod(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot"
            _write_pilot(pilot, sparse_l2=0.1)

            summary = run_norm_budgeted_churn_strata_synthesis(
                pilot_dir=pilot,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "norm_budgeted_churn_strata_synthesis_completed")
            self.assertFalse(summary["runpod_repeat_warranted"])
            self.assertFalse(summary["requires_gpu_now"])
            with (root / "out" / "arm_signal_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            sparse = next(row for row in rows if row["arm"] == "sparse_contextual_topk2_norm_budgeted")
            self.assertEqual(sparse["scientific_signal"], "blocked_or_control")
            self.assertEqual(sparse["blocker"], "residual_l2_fraction_below_nontrivial_budget")
            with (root / "out" / "strata_summary.csv").open(newline="", encoding="utf-8") as handle:
                strata = list(csv.DictReader(handle))
            self.assertTrue(strata)
            self.assertIn("residual_l2_bin", strata[0])
            self.assertIn("anchor_kl_bin", strata[0])
            self.assertIn("base_loss_bin", strata[0])

    def test_nontrivial_matched_sparse_signal_is_marked_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot"
            _write_pilot(pilot, sparse_l2=0.9)

            summary = run_norm_budgeted_churn_strata_synthesis(
                pilot_dir=pilot,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertTrue(summary["runpod_repeat_warranted"])
            self.assertFalse(summary["promotion_allowed"])
            sparse = next(
                row for row in summary["arm_signal_summary"] if row["arm"] == "sparse_contextual_topk2_norm_budgeted"
            )
            self.assertEqual(sparse["scientific_signal"], "weak_local_signal_needs_repeat")

    def test_explicit_churn_objective_interference_stops_local_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot"
            _write_pilot(pilot, sparse_l2=0.9, sparse_anchor_delta=0.01, sparse_flip_delta=0.02, explicit_churn=True)

            summary = run_norm_budgeted_churn_strata_synthesis(
                pilot_dir=pilot,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertFalse(summary["runpod_repeat_warranted"])
            self.assertTrue(summary["explicit_churn_anchor_objective_present"])
            self.assertIn("pivot to a task-free continual-learning", summary["selected_next_step"])
            self.assertIn("did not remove the sparse top-k2 interference", summary["interpretation"])


def _write_pilot(
    path: Path,
    *,
    sparse_l2: float,
    sparse_anchor_delta: float = -0.01,
    sparse_flip_delta: float = -0.05,
    explicit_churn: bool = False,
) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"status": "pass"}) + "\n", encoding="utf-8")
    explicit_header = (
        ",churn_anchor_kl_weight,churn_flip_margin_weight,train_off_target_anchor_kl_trajectory,train_flip_margin_penalty_trajectory"
        if explicit_churn
        else ""
    )
    dense_explicit = ",1.5,0.75,0.001;0.002,0.001;0.002" if explicit_churn else ""
    sparse_explicit = ",1.5,0.75,0.003;0.004,0.003;0.004" if explicit_churn else ""
    (path / "arm_metrics.csv").write_text(
        "arm,family,heldout_ce_loss,heldout_residual_update_l2,ce_delta_vs_dense24,anchor_kl_delta_vs_dense24,flip_delta_vs_dense24"
        f"{explicit_header}\n"
        f"dense_rank24_norm_budgeted,dense_control,3.6,1.0,0.0,0.0,0.0{dense_explicit}\n"
        f"sparse_contextual_topk2_norm_budgeted,sparse_acsr,3.5,{sparse_l2},-0.1,{sparse_anchor_delta},{sparse_flip_delta}{sparse_explicit}\n",
        encoding="utf-8",
    )
    rows = [
        ["dense_rank24_norm_budgeted", "dense_control", "target_heldout", "3.6", "3.7", "-0.1", "0.9", "0.0005", "False"],
        ["dense_rank24_norm_budgeted", "dense_control", "off_target_anchor", "3.5", "3.6", "-0.1", "0.9", "0.0005", "False"],
        [
            "sparse_contextual_topk2_norm_budgeted",
            "sparse_acsr",
            "target_heldout",
            "3.4",
            "3.7",
            "-0.3",
            str(sparse_l2),
            "0.0005",
            "False",
        ],
        [
            "sparse_contextual_topk2_norm_budgeted",
            "sparse_acsr",
            "off_target_anchor",
            "3.4",
            "3.6",
            "-0.2",
            str(sparse_l2),
            "0.0005",
            "False",
        ],
    ]
    with (path / "per_token_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "arm",
                "family",
                "intervention_stratum",
                "ce_loss",
                "base_ce_loss",
                "delta_vs_base_ce",
                "residual_update_l2",
                "anchor_kl_vs_base",
                "prediction_changed_vs_base",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
