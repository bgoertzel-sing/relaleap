from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_gate_report import (
    CAUSAL_GATE_PREREGISTERED,
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    run_causal_contextual_router_gate_report,
)


class CausalContextualRouterGateReportTest(unittest.TestCase):
    def test_preregisters_gate_and_reclassifies_full_context_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence = root / "sequence.json"
            _write_json(sequence, _sequence_packet())
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Reclassify the full-context contextual router as a nondeployable oracle/diagnostic baseline",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_gate_report(
                sequence_report_path=sequence,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], CAUSAL_GATE_PREREGISTERED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertTrue(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(
                summary["claim_statuses"]["contextual_mlp"],
                "nondeployable_full_context_oracle_diagnostic_only",
            )
            self.assertEqual(
                summary["claim_statuses"]["contextual_mlp_causal"],
                "local_sequence_holdout_candidate_not_promoted",
            )
            gate = summary["local_gate_status"]
            self.assertTrue(gate["passed_without_future_perturbation_test"])
            self.assertFalse(gate["passes_full_local_gate"])
            self.assertEqual(
                gate["missing_required_next_check"],
                "future_perturbation_invariance",
            )
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(dispositions[SELECTED_NEXT_ACTION], "selected")
            self.assertEqual(dispositions["runpod_repeat_matrix_now"], "deferred")
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "candidate_actions.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_sequence_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_causal_contextual_router_gate_report(
                sequence_report_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("contextual_router_sequence_kfold_ablation", "source_artifact"),
                fields,
            )


def _sequence_packet() -> dict[str, object]:
    return {
        "status": "ok",
        "decision": "causal_contextual_router_sequence_holdout_candidate",
        "claim_status": "causal_feature_safe_router_local_sequence_holdout_supported",
        "ablation": {
            "fold_count": 4,
            "future_context_material_loss_delta": 0.1743,
            "promoted_vs_linear_loss_delta": -0.6858,
            "causal_contextual_vs_linear_loss_delta": -0.6511,
            "causal_contextual_vs_promoted_full_loss_delta": 0.0347,
            "variants": {
                "promoted_contextual_topk2:actual_full_context": {
                    "mean_router_loss": 2.8933,
                    "uses_future_context": True,
                    "causal_feature_safe": False,
                },
                "promoted_contextual_topk2:causal_current_past_position": {
                    "mean_router_loss": 3.0676,
                    "uses_future_context": False,
                    "causal_feature_safe": True,
                },
                "causal_contextual_topk2:actual_causal_context": {
                    "mean_router_loss": 2.928,
                    "mean_router_oracle_gap": 0.071,
                    "mean_used_columns": 23.0,
                    "mean_unique_support_sets": 51.5,
                    "uses_future_context": False,
                    "causal_feature_safe": True,
                },
                "linear_topk2_control:linear_actual": {
                    "mean_router_loss": 3.5791,
                    "mean_router_oracle_gap": 0.0335,
                    "mean_used_columns": 11.5,
                    "mean_unique_support_sets": 26.5,
                    "uses_future_context": False,
                    "causal_feature_safe": True,
                },
                "causal_contextual_topk2:current_past_no_position": {
                    "mean_router_loss": 2.9282,
                    "uses_future_context": False,
                    "causal_feature_safe": True,
                },
                "causal_contextual_topk2:position_only": {
                    "mean_router_loss": 4.1643,
                    "uses_future_context": False,
                    "causal_feature_safe": True,
                },
            },
            "key_comparisons": {
                "causal_contextual_vs_linear": {
                    "fold_count": 4,
                    "left_wins": 4,
                    "right_wins": 0,
                    "mean_loss_delta": -0.6511,
                    "fold_deltas": [
                        {"fold": 0, "loss_delta": -0.6487},
                        {"fold": 1, "loss_delta": -0.6617},
                        {"fold": 2, "loss_delta": -0.6154},
                        {"fold": 3, "loss_delta": -0.6787},
                    ],
                },
                "full_context_oracle_baseline_vs_linear": {
                    "fold_count": 4,
                    "left_wins": 4,
                    "right_wins": 0,
                    "mean_loss_delta": -0.6858,
                },
                "causal_contextual_vs_full_context_oracle_baseline": {
                    "fold_count": 4,
                    "left_wins": 0,
                    "right_wins": 4,
                    "mean_loss_delta": 0.0347,
                },
            },
        },
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
