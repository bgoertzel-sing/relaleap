from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.synthetic_mechanism_causal_modularity import (
    REQUIRED_ARTIFACTS,
    run_synthetic_mechanism_causal_modularity,
)


class SyntheticMechanismCausalModularityTest(unittest.TestCase):
    def test_generates_hidden_boundary_packet_and_fails_closed_without_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=out_dir,
                seed=3,
                vocab_size=12,
                seq_len=6,
                train_episodes_per_rule=2,
                holdout_episodes_per_rule=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_pregate_failed_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["notify_ben"])
            self.assertEqual(summary["strategic_change_level"], "major")
            self.assertFalse(summary["task_id_visible_to_model"])
            self.assertFalse(summary["mechanism_labels_enter_training"])
            self.assertGreater(summary["episode_row_count"], 0)
            self.assertGreater(summary["per_mechanism_intervention_row_count"], 0)
            self.assertGreater(summary["commutator_row_count"], 0)
            self.assertGreater(summary["forgetting_row_count"], 0)
            self.assertTrue(summary["missing_training_hooks"])
            failed = {row["criterion"] for row in summary["failures"]}
            self.assertIn("training_hooks_available", failed)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out_dir / artifact).is_file(), artifact)

            with (out_dir / "episode_rows.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["task_id_visible_to_model"] for row in rows}, {"False"})
            self.assertEqual({row["mechanism_label_enters_training"] for row in rows}, {"False"})
            self.assertEqual({row["shared_vocab_id_space"] for row in rows}, {"True"})
            self.assertEqual(
                {row["latent_rule"] for row in rows},
                {"copy_shift", "reverse_window", "xor_prev", "affine_jump"},
            )
            self.assertIn("True", {row["mechanism_boundary_hidden"] for row in rows})

            with (out_dir / "comparator_controls.csv").open(newline="", encoding="utf-8") as handle:
                controls = list(csv.DictReader(handle))
            arms = {row["arm"] for row in controls}
            self.assertIn("promoted_contextual_topk2", arms)
            self.assertIn("intervention_trained_sparse_topk2", arms)
            self.assertIn("dense_rank_norm_matched", arms)
            self.assertIn("low_churn_mlp_control", arms)
            self.assertIn("token_position_router_topk2", arms)

    def test_schema_can_pass_when_training_hooks_are_declared_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=Path(tmpdir) / "out",
                training_hooks_available=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_pregate_ready")
            self.assertFalse(summary["missing_training_hooks"])
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(criteria["training_hooks_available"]["passed"])

    def test_tiny_cpu_training_smoke_wires_required_arms_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=out_dir,
                seed=5,
                vocab_size=12,
                seq_len=5,
                train_episodes_per_rule=1,
                holdout_episodes_per_rule=1,
                run_training_smoke=True,
                training_steps=2,
                hidden_dim=16,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_local_gates_failed_closed")
            self.assertEqual(summary["local_scientific_gate_status"], "fail")
            self.assertTrue(summary["training_smoke_ran"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["arm_metric_row_count"], 8)
            self.assertGreater(summary["per_token_metric_row_count"], 0)
            self.assertFalse(summary["missing_training_hooks"])
            self.assertIn("not causal modularity evidence", summary["training_smoke_primary_result"]["interpretation"])
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(criteria["training_smoke_required_arms_present"]["passed"])
            self.assertTrue(criteria["training_smoke_intervention_metrics_present"]["passed"])
            self.assertTrue(criteria["training_smoke_commutator_metrics_present"]["passed"])
            scientific = {row["criterion"]: row for row in summary["local_scientific_gates"]}
            self.assertTrue(scientific["measured_training_smoke_metrics_present"]["passed"])
            self.assertIn("stored_parameter_budget", scientific)
            self.assertIn("forgetting_and_functional_churn_measured", scientific)
            self.assertTrue(scientific["stored_parameter_budget"]["passed"])
            self.assertTrue(scientific["forgetting_and_functional_churn_measured"]["passed"])
            self.assertTrue((out_dir / "local_scientific_gates.csv").is_file())

            with (out_dir / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in arm_rows}
            self.assertEqual(
                {
                    "base_no_residual",
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "random_support_topk2",
                    "fixed_support_topk2",
                    "token_position_router_topk2",
                    "dense_rank_norm_matched",
                    "low_churn_mlp_control",
                },
                arms,
            )
            sparse_floor = max(
                int(row["stored_parameter_floor"])
                for row in arm_rows
                if row["arm"] == "promoted_contextual_topk2"
            )
            dense_mlp_stored = [
                int(row["stored_parameters"])
                for row in arm_rows
                if row["arm"] in {"dense_rank_norm_matched", "low_churn_mlp_control"}
            ]
            self.assertTrue(dense_mlp_stored)
            self.assertGreaterEqual(max(dense_mlp_stored), sparse_floor)

            with (out_dir / "per_mechanism_interventions.csv").open(newline="", encoding="utf-8") as handle:
                intervention_rows = list(csv.DictReader(handle))
            self.assertTrue(intervention_rows)
            self.assertEqual({row["metric_values_available"] for row in intervention_rows}, {"True"})
            self.assertIn(
                "selected_column_ablation_dropin",
                {row["intervention"] for row in intervention_rows},
            )
            sparse_rows = [
                row
                for row in intervention_rows
                if row["arm"] == "intervention_trained_sparse_topk2"
            ]
            self.assertTrue(any(row["selected_columns"] for row in sparse_rows))
            self.assertIn("necessity", intervention_rows[0])
            self.assertIn("off_target_leakage", intervention_rows[0])

            with (out_dir / "commutator_rows.csv").open(newline="", encoding="utf-8") as handle:
                commutator_rows = list(csv.DictReader(handle))
            self.assertTrue(commutator_rows)
            self.assertTrue(
                any(
                    float(row["finite_update_commutator_l2"]) > 0.0
                    for row in commutator_rows
                    if row["arm"] != "base_no_residual"
                )
            )

            with (out_dir / "forgetting_rows.csv").open(newline="", encoding="utf-8") as handle:
                forgetting_rows = list(csv.DictReader(handle))
            self.assertTrue(forgetting_rows)
            self.assertEqual({row["metric_values_available"] for row in forgetting_rows}, {"True"})
            self.assertTrue(
                any(
                    abs(float(row["forgetting_delta"])) > 0.0
                    or abs(float(row["functional_churn"])) > 0.0
                    for row in forgetting_rows
                    if row["arm"] != "base_no_residual"
                )
            )
            self.assertTrue(all(row["ce_before"] for row in forgetting_rows))
            self.assertTrue(all(row["residual_l2"] for row in forgetting_rows))


if __name__ == "__main__":
    unittest.main()
