from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.sparse_target_adaptation_rescue_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    REQUIRED_ARTIFACTS,
    RETIRE_CURRENT_RESCUE,
    run_sparse_target_adaptation_rescue_closeout_report,
)


class SparseTargetAdaptationRescueCloseoutReportTest(unittest.TestCase):
    def test_closeout_retires_current_rescue_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mechanism = root / "mechanism.json"
            rescue = root / "rescue.json"
            review = root / "latest-review.md"
            _write_mechanism(mechanism)
            _write_rescue(rescue)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Patch slice semantics, then run mechanism CL",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_sparse_target_adaptation_rescue_closeout_report(
                out_dir=root / "closeout",
                mechanism_probe_path=mechanism,
                rescue_probe_path=rescue,
                strategy_review_path=review,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], RETIRE_CURRENT_RESCUE)
            self.assertEqual(
                summary["claim_status"],
                "topk2_value_lr_or_focal_rescue_not_established",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(
                summary["evidence"]["rescue_best_arm"],
                "contextual_topk2_value_lr4_anchor_kl",
            )
            self.assertIn(
                "mechanistically different sparse-retention objective",
                summary["selected_next_step"],
            )
            self.assertEqual(len(summary["closeout_rows"]), 3)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "closeout" / artifact).is_file(), artifact)

    def test_closeout_fails_closed_when_rescue_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mechanism = root / "mechanism.json"
            rescue = root / "missing-rescue.json"
            _write_mechanism(mechanism)

            summary = run_sparse_target_adaptation_rescue_closeout_report(
                out_dir=root / "closeout",
                mechanism_probe_path=mechanism,
                rescue_probe_path=rescue,
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure["source"], failure["field"])
                for failure in summary["failures"]
            }
            self.assertIn(("sparse_target_adaptation_rescue_probe", "source_artifact"), fields)


def _write_mechanism(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "mechanism_factorized_continual_learning_probe_recorded",
            "claim_status": "mechanism_factorized_sparse_retention_not_established",
            "primary_result": {
                "topk1_minus_dense_mean_target_ce_delta": 0.47,
                "topk1_minus_dense_mean_off_target_kl": -1.51,
                "topk1_minus_dense_mean_final_forgetting": -0.21,
            },
        },
    )


def _write_rescue(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "decision": "sparse_target_adaptation_rescue_probe_recorded",
            "claim_status": "sparse_target_adaptation_rescue_not_established",
            "primary_result": {
                "best_rescue_arm": "contextual_topk2_value_lr4_anchor_kl",
                "best_rescue_minus_dense_target_ce_delta": -0.08,
                "best_rescue_minus_topk2_off_target_kl": 0.74,
            },
        },
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
