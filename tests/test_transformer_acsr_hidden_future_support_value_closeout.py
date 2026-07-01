from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_support_value_closeout import (
    CLOSE_ACTION,
    DENSE_TRACK_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_hidden_future_support_value_closeout,
)


class TransformerACSRHiddenFutureSupportValueCloseoutTests(unittest.TestCase):
    def test_closes_teacher_imitation_and_points_to_dense_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            headroom = root / "headroom.json"
            control = root / "control.json"
            pregate = root / "pregate.json"
            selector = root / "selector.json"
            review = root / "latest-review.md"
            _write_json(
                headroom,
                {
                    "status": "pass",
                    "decision": "support_value_headroom_negligible_close_teacher_imitation_before_gpu",
                    "claim_status": "teacher_support_imitation_has_insufficient_same_student_value_headroom",
                    "value_target_training_allowed": False,
                    "train_mean_oracle_router_gap": 0.00005,
                    "heldout_mean_oracle_router_gap": 0.00015,
                    "headroom_threshold": 0.005,
                    "split_summary": [
                        {
                            "split": "heldout",
                            "mean_teacher_router_delta": 0.009,
                            "mean_predicted_router_delta": 0.010,
                        }
                    ],
                },
            )
            _write_json(
                control,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_future_control_audit_gpu_blocked",
                    "same_student_loss_gate_passes": False,
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                pregate,
                {
                    "status": "pass",
                    "null_margin_gate_passes": True,
                    "same_student_loss_gate_passes": False,
                    "prefix_hidden_jaccard": 0.968,
                    "prefix_hidden_mean_forced_minus_student_router_loss": 0.010,
                },
            )
            _write_json(
                selector,
                {
                    "status": "pass",
                    "selected_next_action": DENSE_TRACK_ACTION,
                    "next_step": "run a bounded local dense/MLP mechanism follow-up",
                    "requires_gpu_now": False,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Replace exact commutator with support-value headroom before GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_hidden_future_support_value_closeout(
                headroom_path=headroom,
                control_audit_path=control,
                pregate_path=pregate,
                post_selector_path=selector,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_teacher_support_imitation_closed_before_gpu",
            )
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["ben_should_be_notified"])
            next_actions = [
                row
                for row in summary["candidate_actions"]
                if row["candidate_action"] == DENSE_TRACK_ACTION
            ]
            self.assertEqual(next_actions[0]["disposition"], "next_after_closeout")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_transformer_acsr_hidden_future_support_value_closeout(
                headroom_path=root / "missing-headroom.json",
                control_audit_path=root / "missing-control.json",
                pregate_path=root / "missing-pregate.json",
                post_selector_path=root / "missing-selector.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
