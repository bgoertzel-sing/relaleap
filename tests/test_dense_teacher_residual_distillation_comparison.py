from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_residual_distillation_comparison import (
    PRIMARY_VARIANT,
    run_dense_teacher_residual_distillation_comparison,
)


class DenseTeacherResidualDistillationComparisonTest(unittest.TestCase):
    def test_writes_fail_closed_local_pilot_artifacts(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "run:",
                        "  experiment_id: dense_teacher_test",
                        "  seed: 1",
                        "  max_steps: 2",
                        "data:",
                        "  dataset: tiny_shakespeare_char",
                        "  seq_len: 16",
                        "model:",
                        "  base:",
                        "    layers: 1",
                        "    hidden_dim: 32",
                        "  columns:",
                        "    num_columns: 4",
                        "    atoms_per_column: 2",
                        "    top_k: 2",
                        "    contextual_router_hidden_dim: 16",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            acsr_gate = root / "acsr_gate.json"
            contextual_gate = root / "contextual_gate.json"
            prior_closeout = root / "prior_closeout.json"
            _write_source(acsr_gate, decision="acsr_gate")
            _write_source(contextual_gate, decision="contextual_gate")
            _write_source(prior_closeout, decision="prior_closeout")
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run dense teacher comparison.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_teacher_residual_distillation_comparison(
                config_path=config,
                out_dir=root / "out",
                acsr_gate_path=acsr_gate,
                contextual_gate_path=contextual_gate,
                prior_distillation_closeout_path=prior_closeout,
                strategy_review_path=review,
                max_steps=2,
                teacher_steps=1,
                student_steps=1,
                predictor_steps=1,
                teacher_scales=(1.0, 0.5),
            )

            self.assertIn(summary["status"], {"pass", "fail"})
            self.assertEqual(summary["config_train_steps"], 2)
            self.assertEqual(summary["teacher_steps"], 1)
            self.assertEqual(summary["student_steps"], 1)
            self.assertEqual(summary["teacher_scales"], [1.0, 0.5])
            self.assertEqual(summary["claim_statuses"]["promoted_default_router"], "no_default_change")
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "variant_metrics.csv").is_file())
            self.assertTrue((root / "out" / "support_metrics.csv").is_file())
            self.assertTrue((root / "out" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())
            variants = {row["variant"] for row in summary["variant_rows"]}
            arms = {row.get("arm") for row in summary["variant_rows"]}
            self.assertIn(PRIMARY_VARIANT, variants)
            self.assertIn("token_position_only_predicted_support", variants)
            self.assertIn("shuffled_predicted_support", variants)
            self.assertIn("parameter_matched_causal_mlp_control", arms)
            self.assertIn("promoted_contextual_topk2_ce_mse_distill", arms)
            self.assertIn("promoted_contextual_topk2_mse_only_distill", arms)
            self.assertIn("rank_matched_contextual_topk1", arms)
            self.assertIn("random_support_topk2", arms)
            self.assertIn("fixed_support_topk2", arms)
            self.assertIn("shuffled_teacher_target_topk2", arms)
            self.assertIn("acsr_predicted_future_support_teacher_scale_0p5", variants)
            self.assertIn("promoted_contextual_topk2_ce_mse_distill_teacher_scale_0p5", arms)
            self.assertIn("dense_teacher_calibrated_scale_0p5", arms)
            self.assertTrue(summary["teacher_scale_summaries"])
            teacher = next(
                row
                for row in summary["variant_rows"]
                if row.get("arm") == "parameter_matched_causal_mlp_control"
            )
            self.assertTrue(Path(teacher["teacher_hidden_residual_export"]).is_file())
            self.assertTrue(Path(teacher["teacher_logit_residual_export"]).is_file())
            for field in (
                "stored_params",
                "active_params",
                "active_rank_or_topk",
                "residual_l2",
                "residual_norm_ratio",
                "flops_estimate",
                "teacher_residual_mse",
                "teacher_residual_r2",
                "teacher_residual_cosine",
            ):
                self.assertIn(field, teacher)
            criteria = {row["criterion"] for row in summary["gate_status"]["criteria"]}
            self.assertIn("source_gates_present_and_passing", criteria)
            self.assertIn("acsr_beats_token_position_and_shuffled_distillation_nulls", criteria)
            self.assertIn("calibrated_teacher_scale_gate", criteria)


def _write_source(path: Path, *, decision: str) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": decision,
                "claim_status": "test_source_passed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
