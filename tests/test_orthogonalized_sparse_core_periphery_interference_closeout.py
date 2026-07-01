from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.orthogonalized_sparse_core_periphery_interference_closeout import (
    MULTISITE_PC_ASSAY_ACTION,
    REPAIR_SOURCES_ACTION,
    REQUIRED_ARTIFACTS,
    run_orthogonalized_sparse_core_periphery_interference_closeout,
)


class OrthogonalizedSparseCorePeripheryInterferenceCloseoutTests(unittest.TestCase):
    def test_closes_one_site_branch_and_selects_multisite_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pilot = root / "pilot.json"
            review = root / "latest-review.md"
            _write_json(
                pilot,
                {
                    "status": "pass",
                    "decision": "orthogonalized_sparse_core_periphery_interference_pilot_recorded",
                    "claim_status": "bounded_local_cpu_training_rows_recorded_no_gpu_claim",
                    "scientific_gate": "blocked",
                    "training_rows_present": True,
                    "synthetic_rows_only": False,
                    "observable_gates": [
                        {"criterion": "ce_guardrail", "passed": False},
                        {"criterion": "functional_churn_flip_rate", "passed": False},
                        {"criterion": "periphery_first_pruning_delta", "passed": False},
                        {"criterion": "retention_after_sequential_updates", "passed": True},
                        {"criterion": "finite_update_commutator_symmetric_kl", "passed": True},
                        {"criterion": "intervention_selectivity", "passed": True},
                    ],
                    "arm_metrics": [
                        {
                            "arm": "orthogonalized_sparse_additive_core_periphery",
                            "ce": 1.13,
                            "functional_churn_flip_rate": 0.42,
                            "retention_after_sequential_updates": 0.79,
                            "finite_update_commutator_symmetric_kl": 0.03,
                            "intervention_selectivity": 1.0,
                            "periphery_first_pruning_delta": 0.02,
                        },
                        {
                            "arm": "dense_ridge_residual",
                            "ce": 1.05,
                            "functional_churn_flip_rate": 0.29,
                            "retention_after_sequential_updates": 0.77,
                            "finite_update_commutator_symmetric_kl": 0.07,
                        },
                        {
                            "arm": "random_feature_mlp_residual",
                            "ce": 1.06,
                            "functional_churn_flip_rate": 0.37,
                            "retention_after_sequential_updates": 0.62,
                            "finite_update_commutator_symmetric_kl": 0.16,
                        },
                        {
                            "arm": "orthogonalized_sparse_no_core_protection_ablation",
                            "periphery_first_pruning_delta": 0.01,
                        },
                    ],
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Recover from schema rows, then run a trained pilot.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_orthogonalized_sparse_core_periphery_interference_closeout(
                pilot_path=pilot,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], MULTISITE_PC_ASSAY_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(
                summary["claim_status"],
                "redirect_to_multisite_continual_pc_core_periphery_assay_no_gpu",
            )
            self.assertTrue(summary["evidence"]["ce_or_churn_blocked"])
            self.assertTrue(summary["evidence"]["mechanism_pruning_blocked"])
            self.assertIn("finite_update_commutator_symmetric_kl", summary["evidence"]["positive_diagnostics"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "report" / artifact).is_file(), artifact)

    def test_missing_pilot_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_orthogonalized_sparse_core_periphery_interference_closeout(
                pilot_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_SOURCES_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "report" / "summary.json").is_file())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
