from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_runpod_repeat_decision import (
    INSUFFICIENT_EVIDENCE,
    NEXT_SUPPORT_AUDIT,
    RUNPOD_REPEAT_GATE_PASSED,
    run_causal_contextual_router_runpod_repeat_decision,
)


class CausalContextualRouterRunpodRepeatDecisionTest(unittest.TestCase):
    def test_passes_repeat_gate_and_blocks_default_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token = root / "token.json"
            char = root / "char.json"
            gate = root / "gate.json"
            perturbation = root / "perturbation.json"
            review = root / "latest-review.md"
            _write_json(token, _sequence_packet("tiny_shakespeare_word", -0.6, 0.03))
            _write_json(char, _sequence_packet("tiny_shakespeare_char", -0.7, 0.57))
            _write_json(gate, _gate_packet())
            _write_json(perturbation, _perturbation_packet())
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

            summary = run_causal_contextual_router_runpod_repeat_decision(
                token_sequence_report_path=token,
                char_sequence_report_path=char,
                runpod_gate_report_path=gate,
                future_perturbation_report_path=perturbation,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], RUNPOD_REPEAT_GATE_PASSED)
            self.assertEqual(summary["selected_next_step"], NEXT_SUPPORT_AUDIT)
            self.assertTrue(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(
                summary["claim_statuses"]["contextual_mlp"],
                "nondeployable_full_context_oracle_diagnostic_only",
            )
            self.assertEqual(
                summary["claim_statuses"]["promoted_default"],
                "blocked_pending_causal_support_audit",
            )
            self.assertTrue(summary["repeat_gate_status"]["passes_runpod_repeat_gate"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "repeat_metrics.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_repeat_artifact_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            char = root / "char.json"
            gate = root / "gate.json"
            perturbation = root / "perturbation.json"
            _write_json(char, _sequence_packet("tiny_shakespeare_char", -0.7, 0.57))
            _write_json(gate, _gate_packet())
            _write_json(perturbation, _perturbation_packet())

            summary = run_causal_contextual_router_runpod_repeat_decision(
                token_sequence_report_path=root / "missing.json",
                char_sequence_report_path=char,
                runpod_gate_report_path=gate,
                future_perturbation_report_path=perturbation,
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
                ("runpod_token_larger_sequence_kfold", "source_artifact"),
                fields,
            )


def _sequence_packet(dataset: str, causal_delta: float, causal_vs_full: float) -> dict[str, object]:
    return {
        "status": "ok",
        "decision": "causal_contextual_router_sequence_holdout_candidate",
        "claim_status": "causal_feature_safe_router_local_sequence_holdout_supported",
        "ablation": {
            "dataset": dataset,
            "fold_count": 4,
            "future_context_material_loss_delta": 0.2,
            "causal_contextual_vs_linear_loss_delta": causal_delta,
            "causal_contextual_vs_promoted_full_loss_delta": causal_vs_full,
            "variants": {
                "promoted_contextual_topk2:actual_full_context": {
                    "mean_router_loss": 2.0,
                    "uses_future_context": True,
                    "causal_feature_safe": False,
                },
                "causal_contextual_topk2:actual_causal_context": {
                    "mean_router_loss": 2.5,
                    "mean_used_columns": 21.0,
                    "mean_unique_support_sets": 55.0,
                    "causal_feature_safe": True,
                },
                "linear_topk2_control:linear_actual": {
                    "mean_router_loss": 3.2,
                    "mean_used_columns": 10.0,
                    "mean_unique_support_sets": 14.0,
                },
            },
            "key_comparisons": {
                "causal_contextual_vs_linear": {
                    "fold_count": 4,
                    "left_wins": 4,
                    "right_wins": 0,
                    "mean_loss_delta": causal_delta,
                    "fold_deltas": [
                        {"fold": 0, "loss_delta": causal_delta},
                        {"fold": 1, "loss_delta": causal_delta},
                        {"fold": 2, "loss_delta": causal_delta},
                        {"fold": 3, "loss_delta": causal_delta},
                    ],
                },
                "causal_contextual_vs_full_context_oracle_baseline": {
                    "fold_count": 4,
                    "left_wins": 0,
                    "right_wins": 4,
                    "mean_loss_delta": causal_vs_full,
                },
            },
        },
    }


def _gate_packet() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "causal_contextual_router_local_gate_passed",
        "local_gate_status": {"passes_full_local_gate": True},
    }


def _perturbation_packet() -> dict[str, object]:
    return {
        "status": "pass",
        "decision": "causal_router_future_perturbation_invariant",
        "future_perturbation_invariance": True,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
