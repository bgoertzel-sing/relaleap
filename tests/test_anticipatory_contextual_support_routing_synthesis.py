from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_synthesis import (
    ACSR_LOCAL_SYNTHESIS_RECORDED,
    INSUFFICIENT_EVIDENCE,
    RUNPOD_REPLICATION_WARRANTED,
    run_anticipatory_contextual_support_routing_synthesis,
)


class AnticipatoryContextualSupportRoutingSynthesisTest(unittest.TestCase):
    def test_records_two_seed_replication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            review = root / "latest-review.md"
            _write_packet(seed1, acsr_ce=2.88, causal_ce=3.06, teacher_ce=2.87)
            _write_packet(seed2, acsr_ce=2.90, causal_ce=3.05, teacher_ce=2.90)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Synthesize local ACSR seeds.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_anticipatory_contextual_support_routing_synthesis(
                audit_dirs=(seed1, seed2),
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ACSR_LOCAL_SYNTHESIS_RECORDED)
            self.assertEqual(summary["replication_gate"], RUNPOD_REPLICATION_WARRANTED)
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue(summary["aggregates"]["all_required_gates_pass"])
            self.assertTrue(summary["aggregates"]["all_same_student_beats_token_position"])
            self.assertLess(summary["aggregates"]["mean_acsr_minus_causal_ce_loss"], 0.0)
            self.assertEqual(len(summary["packet_rows"]), 2)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "packet_metrics.csv").is_file())
            self.assertTrue((root / "report" / "gate_metrics.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_packet_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed1 = root / "seed1"
            _write_packet(seed1, acsr_ce=2.88, causal_ce=3.06, teacher_ce=2.87)

            summary = run_anticipatory_contextual_support_routing_synthesis(
                audit_dirs=(seed1, root / "missing"),
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failures = {
                (failure.get("packet"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("seed2", "summary_json"), failures)


def _write_packet(path: Path, *, acsr_ce: float, causal_ce: float, teacher_ce: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "pass",
        "decision": "anticipatory_contextual_support_routing_smoke_completed",
        "config_path": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
        "train_steps": 50,
        "predictor_steps": 80,
        "gates": {
            "future_perturbation_invariance": True,
            "acsr_beats_shuffled_ce": True,
            "acsr_beats_token_position_ce": True,
            "acsr_does_not_worsen_causal_regret": True,
        },
        "failures": [],
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "predictor_metrics.csv").write_text(
        "\n".join(
            [
                "predictor,holdout_r2",
                "mlp_causal,0.93",
                "token_position_only,-0.14",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "router_metrics.csv").write_text(
        "\n".join(
            [
                "variant,ce_loss,oracle_regret",
                f"full_context_contextual_topk2_teacher,{teacher_ce},0.003",
                f"causal_feature_safe_contextual_topk2,{causal_ce},0.19",
                f"acsr_mlp_predicted_future,{acsr_ce},0.01",
                "shuffled_predicted_features,3.57,0.68",
                "token_position_only_predicted_features,3.06,0.17",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "same_student_metrics.csv").write_text(
        "\n".join(
            [
                "comparison,acsr_minus_control_ce_loss",
                "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features,-0.18",
                "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features,-0.70",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "feature_perturbation.csv").write_text(
        "check,passed\nfuture_positions_do_not_change_prefix,True\n",
        encoding="utf-8",
    )
    (path / "retention_churn_metrics.csv").write_text(
        "\n".join(
            [
                "phase,variant,anchor_support_churn_after_transfer,anchor_logit_mse_after_transfer,teacher_support_churn",
                "second_context_transfer,acsr_mlp_predicted_future,0.29,0.013,",
                "second_context_transfer,shuffled_predicted_features,0.70,0.19,",
                "second_context_transfer,token_position_only_predicted_features,0.31,0.03,",
                "fixed_context_teacher_reference,acsr_mlp_predicted_future,,,0.04",
                "fixed_context_teacher_reference,shuffled_predicted_features,,,0.66",
                "fixed_context_teacher_reference,token_position_only_predicted_features,,,0.36",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "notes.md").write_text("# Notes\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
