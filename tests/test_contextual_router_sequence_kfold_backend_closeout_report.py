from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_sequence_kfold_backend_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    SEQUENCE_KFOLD_BACKEND_VALIDATED,
    run_contextual_router_sequence_kfold_backend_closeout_report,
)


class ContextualRouterSequenceKfoldBackendCloseoutReportTest(unittest.TestCase):
    def test_closeout_validates_matching_local_and_runpod_sequence_kfold(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local.json"
            runpod = root / "runpod.json"
            review = root / "latest-review.md"
            _write_summary(local, cuda=False, linear_delta=-0.61)
            _write_summary(runpod, cuda=True, linear_delta=-0.64)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run sequence-heldout K-fold evidence before GPU work",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_contextual_router_sequence_kfold_backend_closeout_report(
                local_summary_path=local,
                runpod_summary_path=runpod,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SEQUENCE_KFOLD_BACKEND_VALIDATED)
            self.assertTrue(
                summary["evidence"]["causal_contextual_beats_linear_both_backends"]
            )
            self.assertTrue(
                summary["evidence"][
                    "full_context_beats_causal_contextual_both_backends"
                ]
            )
            self.assertTrue(summary["evidence"]["runpod_cuda_available"])
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "backend_comparisons.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_closeout_fails_closed_when_runpod_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local.json"
            _write_summary(local, cuda=False, linear_delta=-0.61)

            summary = run_contextual_router_sequence_kfold_backend_closeout_report(
                local_summary_path=local,
                runpod_summary_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("runpod_sequence_kfold", "source_artifact"), fields)


def _write_summary(path: Path, *, cuda: bool, linear_delta: float) -> None:
    summary = {
        "status": "ok",
        "decision": "causal_contextual_router_sequence_holdout_candidate",
        "claim_status": "causal_feature_safe_router_local_sequence_holdout_supported",
        "cuda_available": cuda,
        "ablation": {
            "fold_count": 4,
            "key_comparisons": {
                "causal_contextual_vs_linear": {
                    "mean_loss_delta": linear_delta,
                    "left_wins": 4,
                    "right_wins": 0,
                },
                "causal_contextual_vs_full_context_oracle_baseline": {
                    "mean_loss_delta": 0.035,
                    "left_wins": 0,
                    "right_wins": 4,
                },
                "full_context_oracle_baseline_vs_linear": {
                    "mean_loss_delta": -0.65,
                    "left_wins": 4,
                    "right_wins": 0,
                },
            },
        },
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
