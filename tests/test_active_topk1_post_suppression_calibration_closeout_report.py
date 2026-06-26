from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_context_gate_suppression_calibration_audit import (
    CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED,
)
from relaleap.experiments.active_topk1_post_decomposition_decision_report import (
    COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
)
from relaleap.experiments.active_topk1_post_suppression_calibration_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    TOPK1_DIAGNOSTIC_ONLY_RETURN_TO_TOPK2,
    run_active_topk1_post_suppression_calibration_closeout_report,
)
from relaleap.experiments.active_topk1_runpod_post_decomposition_closeout_report import (
    RUNPOD_POST_DECOMPOSITION_VALIDATED,
)


class ActiveTopk1PostSuppressionCalibrationCloseoutReportTest(unittest.TestCase):
    def test_closeout_returns_to_topk2_when_deployable_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runpod = root / "runpod"
            calibration = root / "calibration"
            review = root / "latest-review.md"
            _write_runpod_closeout(runpod)
            _write_failed_calibration(calibration)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: close out top-k1 calibration",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_active_topk1_post_suppression_calibration_closeout_report(
                runpod_closeout_dir=runpod,
                suppression_calibration_dir=calibration,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], TOPK1_DIAGNOSTIC_ONLY_RETURN_TO_TOPK2)
            self.assertEqual(summary["claim_status"], COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS)
            self.assertEqual(
                summary["selected_next_step"],
                "return_main_architecture_loop_to_contextual_topk2_support_routing",
            )
            self.assertFalse(
                summary["evidence"]["interpretation"]["deployable_calibration_passed"]
            )
            self.assertTrue(
                summary["evidence"]["interpretation"]["deployable_calibration_failed"]
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_closeout_fails_closed_when_sources_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_active_topk1_post_suppression_calibration_closeout_report(
                runpod_closeout_dir=root / "missing_runpod",
                suppression_calibration_dir=root / "missing_calibration",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("runpod_post_decomposition_closeout", "summary_json"), fields)
            self.assertIn(("context_gate_suppression_calibration", "summary_json"), fields)


def _write_runpod_closeout(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": RUNPOD_POST_DECOMPOSITION_VALIDATED,
                "claim_status": COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_failed_calibration(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": CONTEXT_GATE_SUPPRESSION_CALIBRATION_FAILED,
                "claim_status": COLUMN_PLUS_CONTEXT_GATE_HYPOTHESIS,
                "evidence": {
                    "metrics": {
                        "deployable_holdout_net_gain": 0.43,
                        "deployable_gain_minus_ungated": -0.04,
                        "deployable_gain_minus_coverage_matched_random": 0.02,
                        "deployable_retained_gain_fraction": 0.91,
                        "deployable_offcontext_harm_suppression_fraction": 0.13,
                    },
                    "signals": {
                        "deployable_gate_passes_pre_registered_criteria": False,
                        "deployable_holdout_net_gain_positive": True,
                        "deployable_retains_enough_own_context_gain": True,
                        "deployable_beats_coverage_matched_random": False,
                        "deployable_suppresses_offcontext_harm": False,
                        "topk2_reference_present": True,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
