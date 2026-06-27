from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_router_regret_churn_failure_inspection import (
    INSUFFICIENT_EVIDENCE,
    INSPECTION_RECORDED,
    run_contextual_router_regret_churn_failure_inspection,
)


class ContextualRouterRegretChurnFailureInspectionTest(unittest.TestCase):
    def test_records_backend_stable_failure_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            runpod = root / "runpod"
            post_sequence = root / "post_sequence.json"
            _write_support_audit(local, linear_regret=0.02, churn=0.36, linear_churn=0.25)
            _write_support_audit(runpod, linear_regret=0.03, churn=0.37, linear_churn=0.24)
            _write_json(
                post_sequence,
                {
                    "status": "pass",
                    "decision": "contextual_router_post_sequence_decision_recorded",
                    "claim_status": "causal_feature_safe_router_not_promoted_support_quality_blocked",
                    "selected_next_step": "run a bounded oracle-regret and functional-churn failure inspection",
                },
            )

            summary = run_contextual_router_regret_churn_failure_inspection(
                local_support_audit_dir=local,
                runpod_support_audit_dir=runpod,
                post_sequence_report_path=post_sequence,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], INSPECTION_RECORDED)
            self.assertEqual(
                summary["claim_status"],
                "causal_router_ce_win_is_not_support_quality_evidence",
            )
            self.assertTrue(summary["evidence"]["all_folds_causal_ce_beats_linear"])
            self.assertTrue(
                summary["evidence"]["all_folds_causal_oracle_frontier_beats_linear"]
            )
            self.assertTrue(
                summary["evidence"]["all_folds_causal_regret_worse_than_linear"]
            )
            self.assertTrue(
                summary["evidence"]["all_folds_causal_churn_worse_than_linear"]
            )
            self.assertEqual(summary["evidence"]["fold_count"], 4)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "fold_failure_deltas.csv").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_runpod_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            post_sequence = root / "post_sequence.json"
            _write_support_audit(local, linear_regret=0.02, churn=0.36, linear_churn=0.25)
            _write_json(post_sequence, {"status": "pass"})

            summary = run_contextual_router_regret_churn_failure_inspection(
                local_support_audit_dir=local,
                runpod_support_audit_dir=root / "missing-runpod",
                post_sequence_report_path=post_sequence,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                ("runpod_support_audit", "source_artifact"),
                {
                    (failure.get("source"), failure.get("field"))
                    for failure in summary["failures"]
                },
            )


def _write_support_audit(
    path: Path, *, linear_regret: float, churn: float, linear_churn: float
) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "causal_contextual_router_support_audit_blocks_promotion",
            "claim_status": "causal_contextual_router_ce_supported_support_quality_not_established",
        },
    )
    (path / "aggregate_metrics.csv").write_text(
        "\n".join(
            [
                (
                    "control,mean_router_loss,mean_oracle_loss,"
                    "mean_oracle_support_regret,mean_functional_churn_logit_l1,"
                    "mean_unique_support_sets,mean_used_columns"
                ),
                f"causal_contextual_topk2,2.9,2.8,0.08,{churn},40,20",
                f"linear_topk2,3.5,3.4,{linear_regret},{linear_churn},15,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        (
            "control,fold,heldout_sequence_index,router_loss,oracle_loss,"
            "oracle_support_regret,functional_churn_logit_l1,unique_support_sets,"
            "used_columns,support_change_fraction"
        )
    ]
    for fold in range(2):
        lines.append(
            f"causal_contextual_topk2,{fold},{fold},2.9,2.8,0.08,{churn},40,20,1.0"
        )
        lines.append(
            f"linear_topk2,{fold},{fold},3.5,3.4,{linear_regret},{linear_churn},15,10,0.8"
        )
    (path / "fold_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
