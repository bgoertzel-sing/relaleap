from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.transformer_acsr_hidden_future_predictor_closeout import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SCALE_CAPTURE_ACTION,
    run_transformer_acsr_hidden_future_predictor_closeout,
)


class TransformerACSRHiddenFuturePredictorCloseoutTests(unittest.TestCase):
    def test_strong_null_margin_but_bad_same_student_loss_selects_local_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate_summary.json"
            controls = root / "control_contract.csv"
            review = root / "latest-review.md"
            _write_json(
                pregate,
                {
                    "status": "pass",
                    "decision": "transformer_acsr_hidden_future_predictor_pregate_gpu_blocked",
                    "claim_status": "prefix_safe_hidden_transformer_does_not_clear_full_mechanism_gate",
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "row_count": 252,
                    "heldout_row_count": 63,
                    "train_sequence_count": 3,
                    "heldout_sequence_count": 1,
                    "prefix_hidden_jaccard": 0.9788,
                    "token_position_jaccard": 0.1693,
                    "shuffled_target_jaccard": 0.1005,
                    "frequency_jaccard": 0.1693,
                    "null_margin_gate_passes": True,
                    "prefix_hidden_mean_forced_minus_student_router_loss": 0.0534,
                    "same_student_loss_gate_passes": False,
                    "downstream_intervention_budget_gate_passes": False,
                },
            )
            _write_controls(controls)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Repair hidden/future capture locally; do not use GPU.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_transformer_acsr_hidden_future_predictor_closeout(
                pregate_path=pregate,
                control_contract_path=controls,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "transformer_acsr_hidden_future_predictor_closeout_gpu_blocked",
            )
            self.assertEqual(summary["selected_next_action"], SCALE_CAPTURE_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "hidden_future_branch_interesting_but_gpu_blocked",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["ben_should_be_notified"])
            self.assertEqual(summary["failures"], [])
            selected = [
                row
                for row in summary["candidate_actions"]
                if row["candidate_action"] == SCALE_CAPTURE_ACTION
                and row["disposition"] == "selected"
            ]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("RunPod and Colab validation remain blocked", notes)

    def test_missing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_transformer_acsr_hidden_future_predictor_closeout(
                pregate_path=root / "missing_summary.json",
                control_contract_path=root / "missing_controls.csv",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
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
            "control": "delayed_previous_teacher_support",
            "status": "available",
            "evidence": "computed from prior prefix target",
        },
        {
            "control": "frequency_support_pair",
            "status": "available",
            "evidence": "computed from train sequences only",
        },
        {
            "control": "exact_arbitrary_pair_same_student_intervention",
            "status": "available",
            "evidence": "scored from exact loss lookup",
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
            "control": "future_perturbation_invariance",
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
