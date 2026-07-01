from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_support_predictor_closeout import (
    CAPTURE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_transformer_acsr_support_predictor_closeout,
)


class TransformerACSRSupportPredictorCloseoutTests(unittest.TestCase):
    def test_negative_support_only_pregate_selects_hidden_future_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate_summary.json"
            controls = root / "control_contract.csv"
            review = root / "latest-review.md"
            _write_json(
                pregate,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_support_predictor_pregate_gpu_blocked",
                    "claim_status": "support_only_prefix_transformer_does_not_clear_full_mechanism_gate",
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "row_count": 756,
                    "heldout_row_count": 189,
                    "prefix_support_jaccard": 0.08994708994708997,
                    "token_position_jaccard": 0.08994708994708997,
                    "shuffled_target_jaccard": 0.09347442680776015,
                    "frequency_jaccard": 0.08994708994708997,
                    "null_margin_gate_passes": False,
                    "downstream_intervention_budget_gate_passes": False,
                },
            )
            _write_controls(controls)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Start a bounded local Transformer-ACSR pregate.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_support_predictor_closeout(
                pregate_path=pregate,
                control_contract_path=controls,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_support_only_branch_closed_hidden_future_capture_required",
            )
            self.assertEqual(summary["selected_next_action"], CAPTURE_ACTION)
            self.assertTrue(summary["support_branch_closed"])
            self.assertTrue(summary["hidden_future_capture_required"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["ben_should_be_notified"])
            self.assertEqual(summary["failures"], [])
            selected = [
                row
                for row in summary["candidate_actions"]
                if row["candidate_action"] == CAPTURE_ACTION
                and row["disposition"] == "selected"
            ]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("RunPod and Colab validation remain blocked", notes)

    def test_missing_pregate_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_transformer_acsr_support_predictor_closeout(
                pregate_path=root / "missing_summary.json",
                control_contract_path=root / "missing_controls.csv",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["support_branch_closed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreaterEqual(len(summary["failures"]), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_controls(path: Path) -> None:
    rows = [
        {
            "control": "token_position_only_transformer",
            "status": "available",
            "evidence": "trained in this report",
        },
        {
            "control": "shuffled_target_transformer",
            "status": "available",
            "evidence": "trained in this report",
        },
        {
            "control": "delayed_previous_support",
            "status": "available",
            "evidence": "computed from prefix-safe previous teacher support",
        },
        {
            "control": "frequency_support_pair",
            "status": "available",
            "evidence": "computed from train folds only",
        },
        {
            "control": "exact_arbitrary_pair_same_student_intervention",
            "status": "missing",
            "evidence": "not present",
        },
        {
            "control": "retention_churn_budget",
            "status": "missing",
            "evidence": "not present",
        },
        {
            "control": "finite_update_commutator_budget",
            "status": "missing",
            "evidence": "not present",
        },
        {
            "control": "hidden_future_chunk_targets",
            "status": "missing",
            "evidence": "not present",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
