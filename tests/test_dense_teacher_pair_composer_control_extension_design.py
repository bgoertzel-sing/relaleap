from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_pair_composer_control_extension_design import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SELECTED_ACTION,
    run_dense_teacher_pair_composer_control_extension_design,
)


class DenseTeacherPairComposerControlExtensionDesignTest(unittest.TestCase):
    def test_missing_truth_audit_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_pair_composer_control_extension_design(
                truth_audit_path=root / "missing-truth-audit.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_local_control_extension_design_from_positive_truth_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            truth_audit = root / "truth-audit.json"
            review = root / "latest-review.md"
            _write_json(truth_audit, _truth_audit_payload())
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run a local true-decoder pair-interaction audit with leakage/null/interference gates before any GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_pair_composer_control_extension_design(
                truth_audit_path=truth_audit,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "dense_teacher_pair_composer_control_extension_design_recorded",
            )
            self.assertEqual(summary["selected_next_action"], SELECTED_ACTION)
            self.assertEqual(summary["claim_status"], "design_only_pair_composer_controls_not_yet_evidence")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["strategy_response"]["recommendation_disposition"], "accepted")
            self.assertFalse(summary["strategy_response"]["ben_should_be_notified"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            control_fields = {row["control_field"] for row in summary["control_extension_contract"]}
            self.assertIn("learned_causal_pair_router", control_fields)
            self.assertIn("finite_update_commutator", control_fields)
            probe_arms = {row["arm"] for row in summary["probe_arms"]}
            self.assertIn("delayed_pair_target_null", probe_arms)
            self.assertIn("matched_mlp_residual", probe_arms)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)


def _truth_audit_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "dense_teacher_pair_composer_truth_audit_gpu_blocked",
        "claim_status": "pair_composer_positive_signal_controls_recorded_but_not_cleared",
        "selected_next_action": "extend_pair_composer_truth_audit_controls_before_gpu",
        "advance_to_gpu_validation": False,
        "promotion_allowed": False,
        "pair_metrics": {
            "pair_beats_independent": True,
            "pair_beats_feature_count_null": True,
            "pair_vs_independent_holdout_ce_gain": 0.8667513728141785,
            "pair_vs_feature_count_null_holdout_ce_gain": 1.9462355971336365,
            "pair_train_holdout_ce_gap": 0.4173409342765808,
            "holdout_token_count": 101,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
