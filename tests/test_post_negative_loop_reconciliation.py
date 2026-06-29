from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.post_negative_loop_reconciliation import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_post_negative_loop_reconciliation,
)


class PostNegativeLoopReconciliationTest(unittest.TestCase):
    def test_records_loop_and_selects_local_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            core = root / "core.json"
            pair = root / "pair.json"
            low = root / "low.json"
            review = root / "latest-review.md"
            _write_json(
                core,
                {
                    "status": "pass",
                    "selected_next_action": "demote_current_core_periphery_mechanism_to_diagnostic_status",
                    "next_step": "redirect to dense controls",
                    "claim_status": "current_core_periphery_mechanism_demoted_no_gpu_or_default_change",
                },
            )
            _write_json(
                pair,
                {
                    "status": "pass",
                    "selected_next_action": "redirect_to_core_periphery_predictive_coding_column_design",
                    "next_step": "resume local core/periphery",
                    "claim_status": "dense_teacher_pair_composer_negative_local_evidence_no_gpu",
                },
            )
            _write_json(
                low,
                {
                    "status": "pass",
                    "scientific_gate": "blocked",
                    "selected_next_step": "return to sparse/core-periphery mechanism work",
                    "claim_status": "low_churn_mlp_no_budgeted_advancement_claim",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement local low-churn MLP pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_post_negative_loop_reconciliation(
                core_closeout_path=core,
                pair_closeout_path=pair,
                low_churn_path=low,
                strategy_review_path=review,
                urgent_review_status="timeout",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertEqual(summary["claim_status"], "negative_loop_reconciled_no_gpu_or_promotion")
            self.assertEqual(summary["strategy_response"]["disposition"], "deferred_after_timeout")
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], NEXT_ACTION)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            low = root / "low.json"
            _write_json(low, {"status": "pass", "scientific_gate": "blocked"})

            summary = run_post_negative_loop_reconciliation(
                core_closeout_path=root / "missing-core.json",
                pair_closeout_path=root / "missing-pair.json",
                low_churn_path=low,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
