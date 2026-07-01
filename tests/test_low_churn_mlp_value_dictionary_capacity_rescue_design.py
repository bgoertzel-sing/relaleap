from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_value_dictionary_capacity_rescue_design import (
    IMPLEMENT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_value_dictionary_capacity_rescue_design,
)


class LowChurnMlpValueDictionaryCapacityRescueDesignTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_design(
                closeout_path=root / "missing_closeout.json",
                decision_audit_path=root / "missing_audit.json",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_value_dictionary_rescue_design_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closeout = root / "closeout.json"
            audit = root / "audit.json"
            _write_json(
                closeout,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_sparse_factorization_vector_centroid_ceiling_closed",
                    "claim_status": "current_sparse_factorization_ceiling_closed_value_dictionary_rescue_selected",
                    "selected_next_action": "design_value_dictionary_capacity_rescue_before_gpu",
                    "requires_gpu_now": False,
                    "promotion_allowed": False,
                    "advance_to_gpu_validation": False,
                },
            )
            _write_json(
                audit,
                {
                    "status": "pass",
                    "decision": "low_churn_mlp_sparse_factorization_decision_audit_recorded",
                    "advance_to_gpu_validation": False,
                    "global_dictionary_oracle_r2": 0.409,
                    "covariance_summary": {"effective_dim_90pct": 7, "effective_dim_95pct": 9},
                    "global_dictionary_metrics": [{"support_load_max_fraction": 0.77}],
                },
            )

            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_design(
                closeout_path=closeout,
                decision_audit_path=audit,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], IMPLEMENT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            designs = {row["design"] for row in summary["dictionary_designs"]}
            self.assertIn("multi_codebook_residual_dictionary", designs)
            self.assertIn("low_rank_codebook_dictionary", designs)
            controls = {row["control"] for row in summary["control_rows"]}
            self.assertIn("dense_ridge_same_rows", controls)
            self.assertIn("low_rank_svd_same_rank_sweep", controls)
            self.assertIn("route_scrambled_dictionary", controls)
            gates = {row["gate"] for row in summary["target_noncolumnability_gates"]}
            self.assertIn("dense_low_rank_advantage_margin", gates)
            self.assertIn("richer_oracle_dictionary_min_r2", gates)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("does not promote sparse columns", notes)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
