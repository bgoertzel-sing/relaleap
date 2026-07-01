from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.context_contrastive_core_periphery_closeout import (
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    SPARSE_FACTOR_ACTION,
    run_context_contrastive_core_periphery_closeout,
)


class ContextContrastiveCorePeripheryCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_context_contrastive_core_periphery_closeout(
                probe_path=root / "missing_probe.json",
                low_churn_pilot_path=root / "missing_low_churn.json",
                strategy_review_path=root / "missing_review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_major_strategy_pivot_selects_sparse_factorization_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probe = root / "probe.json"
            low_churn = root / "low_churn.json"
            review = root / "latest-review.md"
            _write_json(
                probe,
                {
                    "status": "pass",
                    "decision": "context_contrastive_core_periphery_probe_recorded_but_blocked",
                    "claim_status": "context_contrastive_core_periphery_not_established",
                    "selected_next_action": "close_or_redesign_context_contrastive_core_periphery_before_gpu",
                    "advance_to_gpu_validation": False,
                    "candidate_observables": {"heldout_ce": 3.79},
                    "failures": [{"criterion": "finite_update_commutator_budget_nonworse"}],
                },
            )
            _write_json(
                low_churn,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_residual_control_pilot_completed",
                    "claim_status": "low_churn_mlp_no_budgeted_advancement_claim",
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                    "gate_criteria": [
                        {
                            "criterion": "budget_gates_fail_closed",
                            "actual": [{"arm": "low_churn_mlp_residual_control", "heldout_ce_loss": 3.61}],
                        }
                    ],
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Stop GPU path and implement a local low-churn-MLP sparse-factorization ceiling.",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_context_contrastive_core_periphery_closeout(
                probe_path=probe,
                low_churn_pilot_path=low_churn,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], SPARSE_FACTOR_ACTION)
            self.assertTrue(summary["ben_should_be_notified"])
            self.assertTrue(summary["direction_shift_recorded"])
            self.assertFalse(summary["requires_gpu_now"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("major direction shift", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
