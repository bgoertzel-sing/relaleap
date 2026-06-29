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
            self.assertFalse(summary["notify_ben"])
            self.assertEqual(summary["strategic_change_level"], "minor")
            self.assertFalse(summary["task_id_visible_to_model"])
            self.assertFalse(summary["mechanism_labels_enter_training"])
            self.assertGreater(summary["episode_row_count"], 0)
            self.assertGreater(summary["per_mechanism_intervention_row_count"], 0)
            self.assertGreater(summary["commutator_row_count"], 0)
            self.assertGreater(summary["forgetting_row_count"], 0)
            self.assertEqual(summary["oracle_support_sparse_topk2_row_count"], 0)
            self.assertEqual(summary["router_value_regret_decomposition_row_count"], 0)
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
            self.assertIn("low_churn_mlp_active_matched", arms)
            self.assertIn("dense_stored_parameter_matched", arms)
            self.assertIn("low_churn_mlp_stored_parameter_matched", arms)
            self.assertIn("token_position_router_topk2", arms)

    def test_schema_can_pass_when_training_hooks_are_declared_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=Path(tmpdir) / "out",
                training_hooks_available=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "synthetic_mechanism_causal_modularity_local_diagnostics_ready_no_promotion")
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
            self.assertIn(
                summary["decision"],
                {
                    "synthetic_mechanism_causal_modularity_local_gates_failed_closed",
                    "synthetic_mechanism_causal_modularity_active_matched_passed_stored_upper_bound_blocks_promotion",
                    "synthetic_mechanism_causal_modularity_local_diagnostics_ready_no_promotion",
                },
            )
            self.assertIn(summary["local_scientific_gate_status"], {"fail", "pass"})
            self.assertIn(summary["active_matched_local_gate_status"], {"fail", "pass"})
            self.assertIn(summary["stored_upper_bound_gap_status"], {"fail", "pass"})
            self.assertTrue(summary["training_smoke_ran"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["arm_metric_row_count"], 10)
            self.assertEqual(summary["ce_gap_decomposition_row_count"], 10)
            self.assertGreater(summary["oracle_support_sparse_topk2_row_count"], 0)
            self.assertIsNotNone(summary["oracle_support_primary_result"])
            self.assertEqual(
                summary["oracle_support_primary_result"]["row_count"],
                summary["oracle_support_sparse_topk2_row_count"],
            )
            self.assertGreater(summary["router_value_regret_decomposition_row_count"], 0)
            self.assertIsNotNone(summary["router_value_regret_primary_result"])
            self.assertEqual(
                summary["router_value_regret_primary_result"]["row_count"],
                summary["router_value_regret_decomposition_row_count"],
            )
            self.assertGreater(summary["per_token_metric_row_count"], 0)
            self.assertGreater(summary["ce_by_rule_position_row_count"], 0)
            self.assertEqual(summary["residual_budget_accounting_row_count"], 10)
            self.assertIsNotNone(summary["residual_budget_primary_result"])
            self.assertFalse(summary["missing_training_hooks"])
            self.assertIn("not causal modularity evidence", summary["training_smoke_primary_result"]["interpretation"])
            criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(criteria["training_smoke_required_arms_present"]["passed"])
            self.assertTrue(criteria["training_smoke_ce_by_rule_position_present"]["passed"])
            self.assertTrue(criteria["training_smoke_residual_budget_accounting_present"]["passed"])
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
                    "low_churn_mlp_active_matched",
                    "dense_stored_parameter_matched",
                    "low_churn_mlp_stored_parameter_matched",
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
                if row["arm"] in {
                    "dense_stored_parameter_matched",
                    "low_churn_mlp_stored_parameter_matched",
                }
            ]
            self.assertTrue(dense_mlp_stored)
            self.assertGreaterEqual(max(dense_mlp_stored), sparse_floor)

            with (out_dir / "ce_gap_decomposition.csv").open(newline="", encoding="utf-8") as handle:
                ce_gap_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ce_gap_rows), 10)
            ce_gap_by_arm = {row["arm"]: row for row in ce_gap_rows}
            self.assertEqual(set(ce_gap_by_arm), arms)
            for required_field in {
                "holdout_ce",
                "residual_l2",
                "active_parameters_proxy",
                "stored_parameters",
                "active_to_best_sparse_ratio",
                "stored_to_best_sparse_ratio",
                "best_sparse_arm",
                "best_dense_mlp_arm",
                "best_active_matched_dense_mlp_arm",
                "best_stored_matched_dense_mlp_arm",
                "best_sparse_ce_minus_best_dense_mlp_ce",
                "best_sparse_ce_minus_best_active_matched_dense_mlp_ce",
                "best_sparse_ce_minus_best_stored_matched_dense_mlp_ce",
                "control_budget_role",
            }:
                self.assertIn(required_field, ce_gap_rows[0])
            self.assertEqual(
                ce_gap_by_arm["dense_rank_norm_matched"]["control_budget_role"],
                "active_proxy_matched_dense_mlp_control",
            )
            self.assertEqual(
                ce_gap_by_arm["dense_stored_parameter_matched"]["control_budget_role"],
                "stored_parameter_matched_dense_mlp_upper_bound",
            )
            self.assertTrue(ce_gap_by_arm["promoted_contextual_topk2"]["best_sparse_arm"])
            self.assertTrue(ce_gap_by_arm["promoted_contextual_topk2"]["best_dense_mlp_arm"])

            with (out_dir / "ce_by_rule_position.csv").open(newline="", encoding="utf-8") as handle:
                ce_rule_position_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ce_rule_position_rows), summary["ce_by_rule_position_row_count"])
            self.assertEqual(
                {row["latent_rule"] for row in ce_rule_position_rows},
                {"copy_shift", "reverse_window", "xor_prev", "affine_jump"},
            )
            for required_field in {
                "arm",
                "latent_rule",
                "position_index",
                "token_count",
                "mean_ce_loss",
                "min_ce_loss",
                "max_ce_loss",
                "accuracy",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, ce_rule_position_rows[0])
            self.assertEqual(
                {row["mechanism_labels_used_for_scoring_only"] for row in ce_rule_position_rows},
                {"True"},
            )

            with (out_dir / "residual_budget_accounting.csv").open(newline="", encoding="utf-8") as handle:
                budget_accounting_rows = list(csv.DictReader(handle))
            self.assertEqual(len(budget_accounting_rows), 10)
            budget_by_arm = {row["arm"]: row for row in budget_accounting_rows}
            self.assertEqual(set(budget_by_arm), arms)
            for required_field in {
                "residual_l2",
                "residual_l2_ratio_vs_best_sparse",
                "active_parameters_proxy",
                "stored_parameters",
                "flop_proxy_per_token",
                "active_to_best_sparse_ratio",
                "stored_to_best_sparse_ratio",
                "flop_to_best_active_dense_mlp_ratio",
                "flop_to_best_stored_dense_mlp_ratio",
                "accounting_is_proxy",
                "flop_proxy_notes",
            }:
                self.assertIn(required_field, budget_accounting_rows[0])
            self.assertEqual(budget_by_arm["base_no_residual"]["flop_proxy_per_token"], "0.0")
            self.assertEqual({row["accounting_is_proxy"] for row in budget_accounting_rows}, {"True"})
            self.assertTrue(budget_by_arm["promoted_contextual_topk2"]["residual_l2_ratio_vs_best_sparse"])

            with (out_dir / "oracle_support_sparse_topk2.csv").open(newline="", encoding="utf-8") as handle:
                oracle_rows = list(csv.DictReader(handle))
            self.assertEqual(len(oracle_rows), summary["oracle_support_sparse_topk2_row_count"])
            self.assertEqual(
                {row["arm"] for row in oracle_rows},
                {"promoted_contextual_topk2", "intervention_trained_sparse_topk2"},
            )
            for required_field in {
                "learned_support",
                "learned_ce_loss",
                "best_singleton_support",
                "best_singleton_ce_loss",
                "best_pair_support",
                "best_pair_ce_loss",
                "oracle_support",
                "oracle_support_size",
                "oracle_ce_loss",
                "oracle_regret",
                "pair_oracle_regret",
                "best_one_swap_support",
                "best_one_swap_ce_loss",
                "one_swap_regret",
                "one_swap_recovery_fraction",
                "singleton_supports_evaluated",
                "pair_supports_evaluated",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, oracle_rows[0])
            self.assertEqual({row["mechanism_labels_used_for_scoring_only"] for row in oracle_rows}, {"True"})
            self.assertTrue(any(float(row["oracle_regret"]) >= 0.0 for row in oracle_rows))
            self.assertTrue(all(row["oracle_support"] for row in oracle_rows))
            self.assertTrue(all(row["best_one_swap_support"] for row in oracle_rows))

            with (out_dir / "router_value_regret_decomposition.csv").open(newline="", encoding="utf-8") as handle:
                regret_rows = list(csv.DictReader(handle))
            self.assertEqual(len(regret_rows), summary["router_value_regret_decomposition_row_count"])
            self.assertEqual(
                {row["arm"] for row in regret_rows},
                {"promoted_contextual_topk2", "intervention_trained_sparse_topk2"},
            )
            self.assertIn("all", {row["latent_rule"] for row in regret_rows})
            for required_field in {
                "mean_learned_ce_loss",
                "mean_oracle_ce_loss",
                "mean_oracle_regret",
                "max_oracle_regret",
                "positive_oracle_regret_fraction",
                "mean_pair_oracle_regret",
                "mean_one_swap_regret",
                "mean_one_swap_recovery_fraction",
                "oracle_pair_fraction",
                "mean_best_pair_ce_minus_best_singleton_ce",
                "learned_support_matches_oracle_fraction",
                "router_value_status",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, regret_rows[0])
            self.assertEqual(
                {row["mechanism_labels_used_for_scoring_only"] for row in regret_rows},
                {"True"},
            )

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

    def test_teacher_distillation_opt_in_adds_sparse_student_and_shuffled_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"

            summary = run_synthetic_mechanism_causal_modularity(
                out_dir=out_dir,
                seed=7,
                vocab_size=12,
                seq_len=5,
                train_episodes_per_rule=1,
                holdout_episodes_per_rule=1,
                run_training_smoke=True,
                training_steps=2,
                hidden_dim=16,
                include_teacher_distillation=True,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertTrue(summary["teacher_distillation_included"])
            self.assertEqual(summary["teacher_distillation_arm_count"], 2)
            self.assertEqual(summary["arm_metric_row_count"], 12)
            self.assertEqual(summary["ce_gap_decomposition_row_count"], 12)
            self.assertEqual(summary["residual_budget_accounting_row_count"], 12)
            self.assertEqual(summary["router_value_regret_decomposition_row_count"], 15)
            teacher_summary = summary["teacher_distillation_primary_result"]
            self.assertEqual(teacher_summary["row_count"], 2)
            self.assertIsNotNone(teacher_summary["distilled_holdout_ce"])
            self.assertIsNotNone(teacher_summary["shuffled_null_holdout_ce"])
            self.assertIn("hard top-k2", teacher_summary["interpretation"])

            required_criteria = {row["criterion"]: row for row in summary["gate_criteria"]}
            self.assertTrue(required_criteria["training_smoke_required_arms_present"]["passed"])

            with (out_dir / "comparator_controls.csv").open(newline="", encoding="utf-8") as handle:
                controls = list(csv.DictReader(handle))
            control_arms = {row["arm"] for row in controls}
            self.assertIn("dense_teacher_distilled_sparse_topk2", control_arms)
            self.assertIn("shuffled_teacher_distilled_sparse_topk2", control_arms)

            with (out_dir / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            by_arm = {row["arm"]: row for row in arm_rows}
            self.assertEqual(by_arm["dense_teacher_distilled_sparse_topk2"]["teacher_distillation_enabled"], "True")
            self.assertEqual(by_arm["dense_teacher_distilled_sparse_topk2"]["shuffled_teacher_null"], "False")
            self.assertEqual(by_arm["shuffled_teacher_distilled_sparse_topk2"]["shuffled_teacher_null"], "True")
            self.assertTrue(by_arm["dense_teacher_distilled_sparse_topk2"]["teacher_residual_mse"])
            self.assertTrue(by_arm["shuffled_teacher_distilled_sparse_topk2"]["teacher_residual_mse"])

            with (out_dir / "ce_gap_decomposition.csv").open(newline="", encoding="utf-8") as handle:
                ce_gap_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ce_gap_rows), 12)
            self.assertIn(
                "dense_teacher_distilled_sparse_topk2",
                {row["arm"] for row in ce_gap_rows},
            )


if __name__ == "__main__":
    unittest.main()
