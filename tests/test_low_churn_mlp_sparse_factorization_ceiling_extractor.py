from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_ceiling_extractor import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_sparse_factorization_ceiling_extractor,
)


class LowChurnMlpSparseFactorizationCeilingExtractorTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_sparse_factorization_ceiling_extractor(
                design_dir=root / "missing_design",
                low_churn_pilot_dir=root / "missing_pilot",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["training_executed"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_extracts_teacher_rows_and_sparse_ceiling_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design"
            pilot = root / "pilot"
            _write_design(design)
            _write_pilot(pilot)

            summary = run_low_churn_mlp_sparse_factorization_ceiling_extractor(
                design_dir=design,
                low_churn_pilot_dir=pilot,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_sparse_factorization_ceiling_extractor_recorded")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["training_executed"])
            self.assertEqual(summary["teacher_residual_row_count"], 2)
            self.assertEqual(summary["heldout_teacher_residual_row_count"], 1)

            with (root / "out" / "teacher_residual_rows.csv").open(newline="", encoding="utf-8") as handle:
                teacher_rows = list(csv.DictReader(handle))
            self.assertEqual({row["teacher_arm"] for row in teacher_rows}, {"low_churn_mlp_residual_control"})
            self.assertEqual({row["raw_teacher_vector_available"] for row in teacher_rows}, {"False"})
            self.assertIn("teacher_residual_update_l2", teacher_rows[0])

            with (root / "out" / "teacher_budget_rows.csv").open(newline="", encoding="utf-8") as handle:
                budget_rows = list(csv.DictReader(handle))
            budget_metrics = {row["metric"] for row in budget_rows}
            self.assertIn("teacher_heldout_residual_update_l2", budget_metrics)
            self.assertIn("shuffled_null_heldout_ce_loss", budget_metrics)

            with (root / "out" / "support_arm_schema.csv").open(newline="", encoding="utf-8") as handle:
                support_rows = list(csv.DictReader(handle))
            support_arms = {row["arm"] for row in support_rows}
            self.assertIn("oracle_support_sparse_ceiling", support_arms)
            self.assertIn("route_scrambled_same_values", support_arms)
            self.assertEqual({row["training_rows_present"] for row in support_rows}, {"False"})

            with (root / "out" / "factorization_schema.csv").open(newline="", encoding="utf-8") as handle:
                schema_rows = list(csv.DictReader(handle))
            schema_fields = {row["field"] for row in schema_rows}
            self.assertIn("finite_update_commutator", schema_fields)
            self.assertIn("intervention_fingerprint_specificity", schema_fields)


def _write_design(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_ceiling_design_recorded",
                "selected_next_action": "implement_low_churn_mlp_sparse_factorization_ceiling_extractor",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "support_arms.csv").write_text(
        "\n".join(
            [
                "arm,support_type,trainable,budget_match,required_splits,role",
                "oracle_support_sparse_ceiling,oracle,True,budget,heldout,upper",
                "learned_router_sparse_factorization,learned,True,budget,heldout,deployable",
                "token_position_router_sparse_factorization,token_position,False,budget,heldout,shortcut",
                "frequency_support_router_sparse_factorization,frequency,False,budget,heldout,null",
                "random_fixed_support_sparse_factorization,random,False,budget,heldout,null",
                "route_scrambled_same_values,route_scrambled,False,budget,heldout,null",
                "shuffled_teacher_residual_sparse_factorization,shuffled_teacher,False,budget,heldout,null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "observable_rows.csv").write_text(
        "\n".join(
            [
                "metric,family,gate",
                "teacher_residual_reconstruction_mse,quality,lower",
                "teacher_gap_closure_fraction,quality,meaningful",
                "heldout_ce_transfer,quality,guardrail",
                "oracle_support_regret,support,explicit",
                "support_entropy_and_load,support,load",
                "functional_churn_kl_and_flip_rate,interference,beat_controls",
                "anchor_kl,interference,bounded",
                "finite_update_commutator,interference,beat_controls",
                "intervention_fingerprint_specificity,causal,specific",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_pilot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_residual_control_pilot_completed",
                "budgets": {
                    "dense24_residual_l2_ceiling": 1.0,
                    "dense24_flip_churn_ceiling": 0.25,
                    "dense24_anchor_logit_mse_ceiling": 0.02,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "arm_metrics.csv").write_text(
        "\n".join(
            [
                "arm,heldout_ce_loss,heldout_residual_update_l2,heldout_anchor_kl_vs_base,heldout_prediction_flip_rate,active_params,stored_params",
                "low_churn_mlp_residual_control,3.6,0.15,0.0001,0.01,1073,1073",
                "low_churn_mlp_shuffled_target_null,3.7,0.2,0.0002,0.02,1073,1073",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "per_token_metrics.csv").write_text(
        "\n".join(
            [
                "arm,token_index,split,ce_loss,base_ce_loss,delta_vs_base_ce,residual_update_l2,logit_mse_vs_base,anchor_kl_vs_base,prediction_changed_vs_base,raw_intervention_available",
                "low_churn_mlp_residual_control,0,train_anchor,4.0,4.1,-0.1,0.12,0.001,0.0001,False,True",
                "low_churn_mlp_residual_control,1,heldout,3.0,3.2,-0.2,0.15,0.002,0.0002,True,True",
                "low_churn_mlp_shuffled_target_null,1,heldout,3.3,3.2,0.1,0.2,0.003,0.0003,False,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
