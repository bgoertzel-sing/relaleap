from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_residual_control_pilot import (
    LOW_CHURN_ARM,
    REQUIRED_ARTIFACTS,
    SHUFFLED_NULL_ARM,
    SPARSE_ARM,
    run_low_churn_mlp_residual_control_pilot,
)


class LowChurnMlpResidualControlPilotTest(unittest.TestCase):
    def test_missing_pregate_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_residual_control_pilot(
                pregate_dir=root / "missing",
                out_dir=root / "out",
                train_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "low_churn_mlp_residual_control_pilot_failed_closed")
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(
                summary["selected_next_action"],
                "return_to_sparse_core_periphery_mechanism_work",
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_runs_bounded_local_low_churn_pilot_from_pregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate"
            fingerprint = root / "fingerprint"
            sparse = root / "sparse"
            _write_pregate(pregate)
            _write_fingerprint(fingerprint)
            _write_sparse_comparator(sparse)

            summary = run_low_churn_mlp_residual_control_pilot(
                pregate_dir=pregate,
                mlp_fingerprint_dir=fingerprint,
                sparse_comparator_dir=sparse,
                out_dir=root / "out",
                train_steps=1,
                seed=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_residual_control_pilot_completed")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertIn(
                summary["selected_next_action"],
                {
                    "inspect_low_churn_mlp_pilot_per_token_rows_before_gpu",
                    "return_to_sparse_core_periphery_mechanism_work",
                },
            )
            self.assertEqual(summary["arm_count"], 6)
            self.assertGreater(summary["per_token_row_count"], 0)
            self.assertGreater(summary["pareto_row_count"], 0)
            self.assertGreaterEqual(summary["intervention_fingerprint_row_count"], 8)

            with (root / "out" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arm_rows = list(csv.DictReader(handle))
            arms = {row["arm"] for row in arm_rows}
            self.assertIn(LOW_CHURN_ARM, arms)
            self.assertIn(SHUFFLED_NULL_ARM, arms)
            self.assertIn(SPARSE_ARM, arms)
            low_churn = next(row for row in arm_rows if row["arm"] == LOW_CHURN_ARM)
            self.assertIn(low_churn["advancement_gate"], {"advances_local_review_only", "blocked_or_reference"})
            self.assertIn("passes_l2_budget", low_churn)
            self.assertIn("passes_nontrivial_l2_fraction", low_churn)
            self.assertIn("passes_anchor_drift_budget", low_churn)
            self.assertIn("passes_flip_churn_budget", low_churn)
            self.assertTrue(low_churn["train_loss_trajectory"])
            self.assertTrue(low_churn["train_anchor_kl_trajectory"])

            with (root / "out" / "per_token_metrics.csv").open(newline="", encoding="utf-8") as handle:
                token_rows = list(csv.DictReader(handle))
            token_fields = set(token_rows[0])
            self.assertIn("ce_loss", token_fields)
            self.assertIn("residual_update_l2", token_fields)
            self.assertIn("anchor_kl_vs_base", token_fields)
            self.assertIn("prediction_changed_vs_base", token_fields)
            self.assertIn("raw_intervention_available", token_fields)
            self.assertIn(LOW_CHURN_ARM, {row["arm"] for row in token_rows})
            self.assertIn(SPARSE_ARM, {row["arm"] for row in token_rows})

            with (root / "out" / "pareto_rows.csv").open(newline="", encoding="utf-8") as handle:
                pareto_rows = list(csv.DictReader(handle))
            pareto_fields = set(pareto_rows[0])
            self.assertIn("anchor_logit_mse", pareto_fields)
            self.assertIn("flip_churn", pareto_fields)
            self.assertIn("active_params", pareto_fields)

            with (root / "out" / "intervention_fingerprints.csv").open(newline="", encoding="utf-8") as handle:
                fingerprint_rows = list(csv.DictReader(handle))
            self.assertIn(LOW_CHURN_ARM, {row["arm"] for row in fingerprint_rows})
            self.assertIn(SHUFFLED_NULL_ARM, {row["arm"] for row in fingerprint_rows})


def _write_pregate(path: Path) -> None:
    path.mkdir(parents=True)
    payload = {
        "status": "pass",
        "selected_next_action": "implement_low_churn_mlp_residual_control_pilot",
        "budget_rows": [
            {"metric": "dense24_residual_l2_ceiling", "value": 1.0},
            {"metric": "dense24_anchor_logit_mse_ceiling", "value": 0.02},
            {"metric": "dense24_flip_churn_ceiling", "value": 0.25},
            {"metric": "dense24_ce_reference", "value": 3.7},
        ],
    }
    (path / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (path / "pregate_arms.csv").write_text("arm\nlow_churn_mlp_residual_control\n", encoding="utf-8")
    (path / "budget_rows.csv").write_text("metric,value\ndense24_residual_l2_ceiling,1.0\n", encoding="utf-8")


def _write_fingerprint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "scaled_interventions.csv").write_text(
        "\n".join(
            [
                "arm,lambda,ce_loss,delta_vs_base_ce,residual_update_l2,logit_mse_vs_base,prediction_changed_vs_base",
                "dense_rank24_best_norm,1.0,3.7,-0.2,1.0,0.02,0.25",
                "parameter_matched_causal_mlp_control,1.0,2.8,-1.1,4.0,0.3,0.8",
                "parameter_matched_causal_mlp_control,0.25,3.8,-0.1,1.0,0.019,0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sparse_comparator(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "per_token_metrics.csv").write_text(
        "\n".join(
            [
                "arm,token_index,split,base_ce_loss,ce_loss,delta_vs_base_ce,residual_update_l2,logit_mse_vs_base,prediction_changed_vs_base,base_logits,candidate_logits,residual_update_vector",
                "sparse_contextual_topk2,0,heldout,4.0,3.8,-0.2,0.9,0.01,False,[0],[1],[0.1]",
                "sparse_contextual_topk2,1,heldout,4.2,3.9,-0.3,0.8,0.02,True,[0],[1],[0.1]",
                "sparse_contextual_topk2,2,train,4.1,4.0,-0.1,0.7,0.01,False,[0],[1],[0.1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
