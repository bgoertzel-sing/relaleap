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
            self.assertEqual(summary["router_regret_ceiling_budget_row_count"], 0)
            self.assertEqual(summary["support_head_sequence_heldout_diagnostic_row_count"], 0)
            self.assertEqual(summary["router_only_branch_selection_row_count"], 0)
            self.assertEqual(summary["teacher_distillation_closeout_row_count"], 0)
            self.assertEqual(summary["value_capacity_core_periphery_diagnostic_row_count"], 0)
            self.assertEqual(summary["core_periphery_sparse_value_capacity_probe_row_count"], 0)
            self.assertEqual(summary["core_periphery_update_stability_bracket_row_count"], 0)
            self.assertEqual(summary["core_periphery_branch_closeout_row_count"], 0)
            self.assertEqual(summary["sparse_value_redesign_selector_row_count"], 0)
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
            self.assertEqual(summary["arm_metric_row_count"], 16)
            self.assertEqual(summary["ce_gap_decomposition_row_count"], 16)
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
            self.assertEqual(summary["router_regret_ceiling_budget_row_count"], 3)
            self.assertIsNotNone(summary["router_regret_ceiling_budget_primary_result"])
            self.assertEqual(
                summary["router_regret_ceiling_budget_primary_result"]["row_count"],
                summary["router_regret_ceiling_budget_row_count"],
            )
            self.assertEqual(summary["support_head_sequence_heldout_diagnostic_row_count"], 12)
            self.assertIsNotNone(summary["support_head_sequence_heldout_diagnostic_primary_result"])
            self.assertEqual(
                summary["support_head_sequence_heldout_diagnostic_primary_result"]["row_count"],
                summary["support_head_sequence_heldout_diagnostic_row_count"],
            )
            self.assertEqual(summary["router_only_branch_selection_row_count"], 3)
            self.assertIsNotNone(summary["router_only_branch_selection_primary_result"])
            self.assertEqual(
                summary["router_only_branch_selection_primary_result"]["row_count"],
                summary["router_only_branch_selection_row_count"],
            )
            self.assertFalse(summary["router_only_branch_selection_primary_result"]["requires_gpu_now"])
            self.assertFalse(summary["router_only_branch_selection_primary_result"]["promotion_allowed"])
            self.assertEqual(summary["value_capacity_core_periphery_diagnostic_row_count"], 3)
            self.assertIsNotNone(summary["value_capacity_core_periphery_diagnostic_primary_result"])
            self.assertEqual(
                summary["value_capacity_core_periphery_diagnostic_primary_result"]["row_count"],
                summary["value_capacity_core_periphery_diagnostic_row_count"],
            )
            self.assertFalse(
                summary["value_capacity_core_periphery_diagnostic_primary_result"]["requires_gpu_now"]
            )
            self.assertFalse(
                summary["value_capacity_core_periphery_diagnostic_primary_result"]["promotion_allowed"]
            )
            self.assertEqual(summary["core_periphery_sparse_value_capacity_probe_row_count"], 4)
            self.assertIsNotNone(summary["core_periphery_sparse_value_capacity_probe_primary_result"])
            self.assertEqual(
                summary["core_periphery_sparse_value_capacity_probe_primary_result"]["row_count"],
                summary["core_periphery_sparse_value_capacity_probe_row_count"],
            )
            self.assertFalse(
                summary["core_periphery_sparse_value_capacity_probe_primary_result"]["advance_to_gpu_validation"]
            )
            self.assertFalse(
                summary["core_periphery_sparse_value_capacity_probe_primary_result"]["requires_gpu_now"]
            )
            self.assertFalse(
                summary["core_periphery_sparse_value_capacity_probe_primary_result"]["promotion_allowed"]
            )
            self.assertEqual(summary["core_periphery_update_stability_bracket_row_count"], 2)
            self.assertIsNotNone(summary["core_periphery_update_stability_bracket_primary_result"])
            self.assertEqual(
                summary["core_periphery_update_stability_bracket_primary_result"]["row_count"],
                summary["core_periphery_update_stability_bracket_row_count"],
            )
            self.assertFalse(
                summary["core_periphery_update_stability_bracket_primary_result"]["requires_gpu_now"]
            )
            self.assertFalse(
                summary["core_periphery_update_stability_bracket_primary_result"]["promotion_allowed"]
            )
            self.assertEqual(summary["core_periphery_branch_closeout_row_count"], 1)
            self.assertIsNotNone(summary["core_periphery_branch_closeout_primary_result"])
            self.assertEqual(
                summary["core_periphery_branch_closeout_primary_result"]["row_count"],
                summary["core_periphery_branch_closeout_row_count"],
            )
            self.assertFalse(
                summary["core_periphery_branch_closeout_primary_result"]["requires_gpu_now"]
            )
            self.assertFalse(
                summary["core_periphery_branch_closeout_primary_result"]["promotion_allowed"]
            )
            self.assertEqual(summary["sparse_value_redesign_selector_row_count"], 3)
            self.assertIsNotNone(summary["sparse_value_redesign_selector_primary_result"])
            self.assertEqual(
                summary["sparse_value_redesign_selector_primary_result"]["row_count"],
                summary["sparse_value_redesign_selector_row_count"],
            )
            self.assertEqual(
                summary["sparse_value_redesign_selector_primary_result"]["selected_candidate_path"],
                "budget_normalized_gated_low_rank_value_mixture",
            )
            self.assertFalse(
                summary["sparse_value_redesign_selector_primary_result"]["requires_gpu_now"]
            )
            self.assertFalse(
                summary["sparse_value_redesign_selector_primary_result"]["promotion_allowed"]
            )
            self.assertGreater(summary["per_token_metric_row_count"], 0)
            self.assertGreater(summary["ce_by_rule_position_row_count"], 0)
            self.assertEqual(summary["residual_budget_accounting_row_count"], 16)
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
                    "core_periphery_sparse_topk2",
                    "flat_column_value_mlp_topk2",
                    "core_only_sparse_topk2",
                    "periphery_only_sparse_topk2",
                    "core_periphery_stability_slow_core_topk2",
                    "flat_column_value_mlp_anchor_topk2",
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
            self.assertEqual(len(ce_gap_rows), 16)
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

            with (out_dir / "core_periphery_sparse_value_capacity_probe.csv").open(newline="", encoding="utf-8") as handle:
                core_periphery_rows = list(csv.DictReader(handle))
            self.assertEqual(len(core_periphery_rows), 4)
            probe_by_arm = {row["arm"]: row for row in core_periphery_rows}
            self.assertEqual(
                set(probe_by_arm),
                {
                    "core_periphery_sparse_topk2",
                    "flat_column_value_mlp_topk2",
                    "core_only_sparse_topk2",
                    "periphery_only_sparse_topk2",
                },
            )
            primary_probe = probe_by_arm["core_periphery_sparse_topk2"]
            for required_field in {
                "probe_role",
                "value_head_variant",
                "ce_gain_vs_reference_sparse",
                "stored_gap_closed_fraction",
                "core_parameter_drift_l2",
                "periphery_l1",
                "residual_norm_clip",
                "residual_norm_clipped",
                "norm_budget_ok",
                "commutator_budget_ok",
                "functional_churn_budget_ok",
                "advance_to_gpu_validation",
                "requires_gpu_now",
                "promotion_allowed",
            }:
                self.assertIn(required_field, primary_probe)
            self.assertEqual(primary_probe["probe_role"], "primary_core_periphery_probe")
            self.assertEqual(primary_probe["residual_norm_clipped"], "True")
            self.assertGreater(float(primary_probe["residual_norm_clip"]), 0.0)
            self.assertEqual(primary_probe["requires_gpu_now"], "False")
            self.assertEqual(primary_probe["promotion_allowed"], "False")

            with (out_dir / "core_periphery_update_stability_bracket.csv").open(newline="", encoding="utf-8") as handle:
                stability_rows = list(csv.DictReader(handle))
            self.assertEqual(len(stability_rows), 2)
            stability_by_arm = {row["arm"]: row for row in stability_rows}
            self.assertEqual(
                set(stability_by_arm),
                {
                    "core_periphery_stability_slow_core_topk2",
                    "flat_column_value_mlp_anchor_topk2",
                },
            )
            for required_field in {
                "bracket_role",
                "unregularized_comparator_arm",
                "ce_delta_vs_unregularized_comparator",
                "commutator_ratio_vs_unregularized_comparator",
                "functional_churn_ratio_vs_unregularized_comparator",
                "anchor_kl_weight",
                "core_lr_scale",
                "core_drift_penalty_weight",
                "stability_candidate",
                "requires_gpu_now",
                "promotion_allowed",
            }:
                self.assertIn(required_field, stability_rows[0])
            self.assertEqual({row["requires_gpu_now"] for row in stability_rows}, {"False"})
            self.assertEqual({row["promotion_allowed"] for row in stability_rows}, {"False"})

            with (out_dir / "core_periphery_branch_closeout.csv").open(newline="", encoding="utf-8") as handle:
                closeout_rows = list(csv.DictReader(handle))
            self.assertEqual(len(closeout_rows), 1)
            closeout = closeout_rows[0]
            for required_field in {
                "closeout_status",
                "primary_arm",
                "primary_ce_gain_vs_reference_sparse",
                "primary_stored_gap_closed_fraction",
                "flat_control_arm",
                "primary_ce_minus_flat_control_ce",
                "flat_control_stronger_by_gt_0p01",
                "primary_budget_passes",
                "primary_signal_passes",
                "update_stability_candidate_count",
                "advance_to_gpu_validation",
                "requires_gpu_now",
                "promotion_allowed",
                "recommend_next_path",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, closeout)
            self.assertEqual(closeout["primary_arm"], "core_periphery_sparse_topk2")
            self.assertEqual(closeout["flat_control_arm"], "flat_column_value_mlp_topk2")
            self.assertIn(
                closeout["closeout_status"],
                {
                    "closed_redesign_required",
                    "repeat_before_closeout",
                    "continue_local_branch",
                },
            )
            self.assertEqual(closeout["advance_to_gpu_validation"], "False")
            self.assertEqual(closeout["requires_gpu_now"], "False")
            self.assertEqual(closeout["promotion_allowed"], "False")
            self.assertEqual(closeout["mechanism_labels_used_for_scoring_only"], "True")

            with (out_dir / "sparse_value_redesign_selector.csv").open(newline="", encoding="utf-8") as handle:
                redesign_rows = list(csv.DictReader(handle))
            self.assertEqual(len(redesign_rows), 3)
            selected_redesigns = [row for row in redesign_rows if row["selected"] == "True"]
            self.assertEqual(len(selected_redesigns), 1)
            selected_redesign = selected_redesigns[0]
            self.assertEqual(
                selected_redesign["candidate_path"],
                "budget_normalized_gated_low_rank_value_mixture",
            )
            for required_field in {
                "source_closeout_status",
                "reference_to_stored_upper_bound_ce_gap",
                "primary_residual_l2_ratio_vs_best_sparse",
                "flat_control_stronger_by_gt_0p01",
                "closed_branch_budget_failure",
                "pregate_required",
                "next_experiment",
                "requires_gpu_now",
                "promotion_allowed",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, selected_redesign)
            self.assertEqual(selected_redesign["source_closeout_status"], "closed_redesign_required")
            self.assertEqual(selected_redesign["requires_gpu_now"], "False")
            self.assertEqual(selected_redesign["promotion_allowed"], "False")
            self.assertEqual(selected_redesign["mechanism_labels_used_for_scoring_only"], "True")

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
            self.assertEqual(len(budget_accounting_rows), 16)
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
                {
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "core_periphery_sparse_topk2",
                },
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
                {
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "core_periphery_sparse_topk2",
                },
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

            with (out_dir / "router_regret_ceiling_budget.csv").open(newline="", encoding="utf-8") as handle:
                ceiling_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ceiling_rows), summary["router_regret_ceiling_budget_row_count"])
            self.assertEqual(
                {row["arm"] for row in ceiling_rows},
                {
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "core_periphery_sparse_topk2",
                },
            )
            for required_field in {
                "learned_holdout_ce",
                "oracle_support_ce_ceiling",
                "oracle_support_ce_gain",
                "token_position_null_ce",
                "learned_ce_gap_to_token_position_null",
                "router_only_can_close_token_position_gap",
                "active_matched_control_arm",
                "active_matched_control_ce",
                "learned_ce_gap_to_active_matched_control",
                "router_only_can_close_active_matched_gap",
                "stored_matched_control_arm",
                "stored_matched_control_ce",
                "learned_ce_gap_to_stored_matched_control",
                "router_only_can_close_stored_matched_gap",
                "router_only_sufficiency_status",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, ceiling_rows[0])
            self.assertEqual(
                {row["mechanism_labels_used_for_scoring_only"] for row in ceiling_rows},
                {"True"},
            )
            self.assertTrue(all(float(row["oracle_support_ce_gain"]) >= -1e-8 for row in ceiling_rows))
            self.assertTrue(all(row["router_only_sufficiency_status"] for row in ceiling_rows))

            with (out_dir / "support_head_sequence_heldout_diagnostic.csv").open(newline="", encoding="utf-8") as handle:
                support_head_rows = list(csv.DictReader(handle))
            self.assertEqual(len(support_head_rows), summary["support_head_sequence_heldout_diagnostic_row_count"])
            self.assertEqual(
                {row["arm"] for row in support_head_rows},
                {
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "core_periphery_sparse_topk2",
                },
            )
            self.assertEqual(
                {
                    "support_regret_trained_contextual_router_topk2",
                    "shuffled_oracle_target_null_topk2",
                    "token_position_only_support_head_topk2",
                    "global_modal_support_null_topk2",
                },
                {row["diagnostic"] for row in support_head_rows},
            )
            for required_field in {
                "learned_router_ce",
                "oracle_pair_ce_ceiling",
                "learned_pair_oracle_regret",
                "predicted_support_ce",
                "predicted_support_ce_gain_vs_learned",
                "oracle_pair_regret_recovery_fraction",
                "support_accuracy_vs_oracle_pair",
                "unique_support_sets",
                "support_load_entropy",
                "support_change_fraction",
                "advance_if_gain_gt_0p02_or_recovery_ge_0p5",
                "beats_shuffled_target_null",
                "beats_token_position_null",
                "oracle_targets_enter_auxiliary_training",
                "deployable_training_evidence",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, support_head_rows[0])
            self.assertEqual({row["split"] for row in support_head_rows}, {"sequence_heldout"})
            self.assertEqual({row["deployable_training_evidence"] for row in support_head_rows}, {"False"})
            self.assertEqual({row["mechanism_labels_used_for_scoring_only"] for row in support_head_rows}, {"True"})
            contextual_rows = [
                row
                for row in support_head_rows
                if row["diagnostic"] == "support_regret_trained_contextual_router_topk2"
            ]
            self.assertTrue(contextual_rows)
            self.assertTrue(all(row["beats_shuffled_target_null"] for row in contextual_rows))
            self.assertTrue(all(row["beats_token_position_null"] for row in contextual_rows))

            with (out_dir / "router_only_branch_selection.csv").open(newline="", encoding="utf-8") as handle:
                branch_rows = list(csv.DictReader(handle))
            self.assertEqual(len(branch_rows), summary["router_only_branch_selection_row_count"])
            self.assertEqual(
                {row["arm"] for row in branch_rows},
                {
                    "promoted_contextual_topk2",
                    "intervention_trained_sparse_topk2",
                    "core_periphery_sparse_topk2",
                },
            )
            for required_field in {
                "decision",
                "close_or_deprioritize_router_only_path",
                "recommend_next_path",
                "requires_gpu_now",
                "promotion_allowed",
                "router_only_can_close_stored_gap",
                "oracle_ce_beats_token_position_null",
                "support_head_advances",
                "deployable_training_evidence",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, branch_rows[0])
            self.assertEqual({row["requires_gpu_now"] for row in branch_rows}, {"False"})
            self.assertEqual({row["promotion_allowed"] for row in branch_rows}, {"False"})
            self.assertEqual({row["deployable_training_evidence"] for row in branch_rows}, {"False"})
            self.assertEqual({row["mechanism_labels_used_for_scoring_only"] for row in branch_rows}, {"True"})

            with (out_dir / "value_capacity_core_periphery_diagnostic.csv").open(newline="", encoding="utf-8") as handle:
                value_capacity_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(value_capacity_rows),
                summary["value_capacity_core_periphery_diagnostic_row_count"],
            )
            self.assertEqual(
                {
                    "active_value_capacity_control",
                    "stored_value_capacity_upper_bound",
                    "core_periphery_sparse_design_probe",
                },
                {row["branch"] for row in value_capacity_rows},
            )
            for required_field in {
                "reference_sparse_arm",
                "comparator_arm",
                "reference_sparse_ce",
                "comparator_ce",
                "sparse_ce_gap_to_comparator",
                "comparator_residual_l2_ratio_vs_sparse",
                "comparator_active_ratio_vs_sparse",
                "comparator_stored_ratio_vs_sparse",
                "comparator_flop_ratio_vs_sparse",
                "reference_sparse_mean_commutator_l2",
                "comparator_mean_commutator_l2",
                "reference_sparse_mean_abs_functional_churn",
                "comparator_mean_abs_functional_churn",
                "router_only_path_closed",
                "candidate_status",
                "recommend_next_path",
                "requires_gpu_now",
                "promotion_allowed",
                "mechanism_labels_used_for_scoring_only",
            }:
                self.assertIn(required_field, value_capacity_rows[0])
            self.assertEqual({row["requires_gpu_now"] for row in value_capacity_rows}, {"False"})
            self.assertEqual({row["promotion_allowed"] for row in value_capacity_rows}, {"False"})
            self.assertEqual({row["mechanism_labels_used_for_scoring_only"] for row in value_capacity_rows}, {"True"})
            design_rows = [
                row
                for row in value_capacity_rows
                if row["branch"] == "core_periphery_sparse_design_probe"
            ]
            self.assertEqual(len(design_rows), 1)
            self.assertIn(
                design_rows[0]["recommend_next_path"],
                {
                    "core_periphery_sparse_value_capacity_probe",
                    "repeat_or_repair_local_value_capacity_diagnostic",
                },
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
            self.assertEqual(summary["arm_metric_row_count"], 18)
            self.assertEqual(summary["ce_gap_decomposition_row_count"], 18)
            self.assertEqual(summary["residual_budget_accounting_row_count"], 18)
            self.assertEqual(summary["router_value_regret_decomposition_row_count"], 20)
            self.assertEqual(summary["router_regret_ceiling_budget_row_count"], 4)
            self.assertEqual(summary["support_head_sequence_heldout_diagnostic_row_count"], 16)
            self.assertEqual(summary["router_only_branch_selection_row_count"], 4)
            self.assertEqual(summary["teacher_distillation_closeout_row_count"], 1)
            self.assertEqual(summary["value_capacity_core_periphery_diagnostic_row_count"], 3)
            self.assertEqual(summary["core_periphery_sparse_value_capacity_probe_row_count"], 4)
            self.assertEqual(summary["core_periphery_update_stability_bracket_row_count"], 2)
            self.assertEqual(summary["core_periphery_branch_closeout_row_count"], 1)
            teacher_summary = summary["teacher_distillation_primary_result"]
            self.assertEqual(teacher_summary["row_count"], 2)
            self.assertIsNotNone(teacher_summary["distilled_holdout_ce"])
            self.assertIsNotNone(teacher_summary["shuffled_null_holdout_ce"])
            self.assertIn("hard top-k2", teacher_summary["interpretation"])
            closeout_summary = summary["teacher_distillation_closeout_primary_result"]
            self.assertEqual(closeout_summary["row_count"], 1)
            self.assertIn(
                closeout_summary["closeout_status"],
                {
                    "closed_non_improving",
                    "closed_teacher_target_not_specific",
                    "router_regret_explains_remaining_gap",
                    "value_distillation_insufficient_vs_dense_controls",
                    "needs_repeat_before_branch_reopen",
                },
            )

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
            self.assertEqual(len(ce_gap_rows), 18)
            self.assertIn(
                "dense_teacher_distilled_sparse_topk2",
                {row["arm"] for row in ce_gap_rows},
            )

            with (out_dir / "teacher_distillation_closeout.csv").open(newline="", encoding="utf-8") as handle:
                closeout_rows = list(csv.DictReader(handle))
            self.assertEqual(len(closeout_rows), 1)
            for required_field in {
                "closeout_status",
                "distilled_holdout_ce",
                "shuffled_null_holdout_ce",
                "distilled_minus_best_sparse_ce",
                "distilled_mean_oracle_regret",
                "router_regret_remains_above_0p02",
                "mechanism_labels_used_for_scoring_only",
                "interpretation",
            }:
                self.assertIn(required_field, closeout_rows[0])
            self.assertEqual(closeout_rows[0]["mechanism_labels_used_for_scoring_only"], "True")


if __name__ == "__main__":
    unittest.main()
