from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_sparse_factorization_ceiling_training_harness import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_sparse_factorization_ceiling_training_harness,
)


class LowChurnMlpSparseFactorizationCeilingTrainingHarnessTests(unittest.TestCase):
    def test_missing_extractor_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_sparse_factorization_ceiling_training_harness(
                extractor_dir=root / "missing",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["runtime_failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_proxy_training_rows_and_blocks_gpu_advancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            extractor = root / "extractor"
            _write_extractor(extractor)

            summary = run_low_churn_mlp_sparse_factorization_ceiling_training_harness(
                extractor_dir=extractor,
                out_dir=root / "out",
                column_count=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_sparse_factorization_ceiling_training_harness_recorded")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertTrue(summary["training_executed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(summary["best_proxy_arm"], "oracle_support_sparse_ceiling")
            self.assertTrue(summary["advancement_failures"])

            with (root / "out" / "training_rows.csv").open(newline="", encoding="utf-8") as handle:
                training_rows = list(csv.DictReader(handle))
            self.assertEqual(len(training_rows), 21)
            self.assertEqual({row["training_mode"] for row in training_rows}, {"scalar_proxy_centroid"})
            self.assertEqual({row["raw_teacher_vector_available"] for row in training_rows}, {"False"})

            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in arm_rows}
            self.assertIn("oracle_support_sparse_ceiling", arms)
            self.assertIn("route_scrambled_same_values", arms)
            self.assertEqual({row["scientific_advancement"] for row in arm_rows}, {"False"})


def _write_extractor(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_ceiling_extractor_recorded",
                "selected_next_action": "implement_low_churn_mlp_sparse_factorization_ceiling_training_harness",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "teacher_residual_rows.csv").write_text(
        "\n".join(
            [
                "teacher_arm,teacher_row_id,token_index,split,base_ce_loss,teacher_ce_loss,teacher_residual_update_l2,raw_teacher_vector_available,raw_intervention_available",
                "low_churn_mlp_residual_control,row0,0,train_anchor,4.0,3.9,0.10,False,True",
                "low_churn_mlp_residual_control,row1,1,train_anchor,4.0,3.8,0.20,False,True",
                "low_churn_mlp_residual_control,row2,2,heldout,4.0,3.7,0.30,False,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "support_arm_schema.csv").write_text(
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
    (path / "teacher_budget_rows.csv").write_text(
        "\n".join(
            [
                "metric,value,role",
                "teacher_heldout_ce_loss,3.6,teacher",
                "shuffled_null_heldout_ce_loss,3.7,null",
                "teacher_heldout_anchor_kl_vs_base,0.001,anchor",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "factorization_schema.csv").write_text(
        "\n".join(
            [
                "field,family,required_for_training_harness,present_in_design,source",
                "teacher_residual_reconstruction_mse,quality,True,True,design",
                "teacher_gap_closure_fraction,quality,True,True,design",
                "finite_update_commutator,interference,True,True,design",
                "intervention_fingerprint_specificity,causal,True,True,design",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
