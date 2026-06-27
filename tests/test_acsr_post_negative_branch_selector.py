from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_post_negative_branch_selector import (
    RETIRE_ACSR_ACTION,
    run_acsr_post_negative_branch_selector,
)


class ACSRPostNegativeBranchSelectorTest(unittest.TestCase):
    def test_selects_retirement_when_common_and_dense_teacher_are_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common.json"
            dense = root / "dense.json"
            review = root / "latest-review.md"
            _write_json(
                common,
                {
                    "status": "fail",
                    "decision": "acsr_common_causal_residual_benchmark_failed_gate",
                    "claim_status": "sparse_support_specific_effect_not_separated_from_common_dense_controls",
                    "failures": [
                        {
                            "criterion": "sparse_beats_causal_dense",
                            "passed": False,
                        }
                    ],
                },
            )
            _write_json(
                dense,
                {
                    "status": "fail",
                    "decision": "dense_teacher_residual_distillation_pilot_not_supported",
                    "claim_status": "dense_teacher_distillation_not_interpretable_or_not_better_than_controls",
                    "failures": [
                        {
                            "criterion": "acsr_ce_not_worse_than_teacher_by_large_margin",
                            "passed": False,
                        }
                    ],
                    "variant_rows": [
                        {
                            "variant": "acsr_predicted_future_support",
                            "ce_loss": 2.8,
                        }
                    ],
                    "dense_teacher_ce_loss": 0.25,
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Freeze ACSR promotion/GPU work",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_post_negative_branch_selector(
                common_benchmark_path=common,
                dense_teacher_path=dense,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], RETIRE_ACSR_ACTION)
            self.assertEqual(summary["claim_statuses"]["ben_notification"], "required")
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], RETIRE_ACSR_ACTION)
            for artifact in ("summary.json", "source_rows.csv", "candidate_actions.csv", "notes.md"):
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_required_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense = root / "dense.json"
            _write_json(dense, {"status": "fail", "decision": "negative"})

            summary = run_acsr_post_negative_branch_selector(
                common_benchmark_path=root / "missing-common.json",
                dense_teacher_path=dense,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertIsNone(summary["selected_next_action"])
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
