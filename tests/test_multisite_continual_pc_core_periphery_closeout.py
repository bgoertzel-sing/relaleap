from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.multisite_continual_pc_core_periphery_closeout import (
    CLOSE_BRANCH_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_multisite_continual_pc_core_periphery_closeout,
)


class MultiSiteContinualPCCorePeripheryCloseoutTests(unittest.TestCase):
    def test_closes_negative_multisite_branch_and_blocks_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            assay = root / "assay"
            assay.mkdir()
            _write_json(
                assay / "summary.json",
                {
                    "status": "pass",
                    "decision": "multisite_continual_pc_core_periphery_assay_recorded",
                    "scientific_gate": "blocked",
                    "claim_status": "trained_local_rows_no_gpu_claim",
                    "training_rows_present": True,
                    "advance_to_gpu_validation": False,
                    "promotion_allowed": False,
                    "primary_result": {
                        "candidate_minus_mlp_heldout_ce": 0.46,
                        "candidate_minus_mlp_churn": 0.02,
                        "candidate_minus_mlp_commutator": -0.04,
                    },
                },
            )
            _write_csv(
                assay / "gate_criteria.csv",
                [
                    {"criterion": "design_contract_passed", "passed": "True", "severity": "hard", "actual": "pass"},
                    {"criterion": "real_training_rows_present", "passed": "True", "severity": "hard", "actual": "12"},
                    {"criterion": "heldout_ce_guardrail", "passed": "False", "severity": "claim", "actual": "2.0"},
                    {
                        "criterion": "functional_churn_no_worse_than_dense_mlp",
                        "passed": "False",
                        "severity": "claim",
                        "actual": "0.53",
                    },
                    {
                        "criterion": "commutator_no_worse_than_dense_mlp",
                        "passed": "False",
                        "severity": "claim",
                        "actual": "0.19",
                    },
                    {"criterion": "cross_site_retention_positive", "passed": "False", "severity": "claim", "actual": "0.39"},
                    {
                        "criterion": "leakage_null_rejection",
                        "passed": "False",
                        "severity": "claim",
                        "actual": "['random_support_sparse_control']",
                    },
                    {"criterion": "no_gpu_promotion", "passed": "True", "severity": "hard", "actual": "advance_to_gpu_validation=false"},
                ],
            )
            _write_csv(
                assay / "arm_metrics.csv",
                [
                    {
                        "arm": "multisite_pc_core_periphery_candidate",
                        "heldout_ce": "2.0",
                        "mean_functional_flip_churn": "0.53",
                        "finite_update_commutator": "0.19",
                        "cross_site_retention": "0.39",
                        "causal_intervention_fingerprint": "0.3",
                        "periphery_first_pruning_delta": "-0.5",
                    },
                    {
                        "arm": "parameter_matched_mlp_residual_control",
                        "heldout_ce": "1.54",
                        "mean_functional_flip_churn": "0.51",
                        "finite_update_commutator": "0.24",
                    },
                    {
                        "arm": "dense_rank_norm_residual_control",
                        "heldout_ce": "1.61",
                        "mean_functional_flip_churn": "0.47",
                        "finite_update_commutator": "0.08",
                    },
                    {"arm": "low_rank_residual_control", "heldout_ce": "1.39"},
                ],
            )
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Keep GPU blocked and use trained local rows.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_multisite_continual_pc_core_periphery_closeout(
                assay_dir=assay,
                strategy_review_path=review,
                out_dir=root / "closeout",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], CLOSE_BRANCH_ACTION)
            self.assertEqual(summary["claim_status"], "multisite_pc_core_periphery_closed_no_gpu")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertIn("heldout_ce_guardrail", summary["evidence"]["failed_claim_gates"])
            self.assertEqual(
                summary["evidence"]["only_positive_candidate_signal"],
                "candidate beats MLP commutator but fails best dense/MLP commutator gate",
            )
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "closeout" / artifact).is_file(), artifact)

    def test_missing_sources_fail_closed_but_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_multisite_continual_pc_core_periphery_closeout(
                assay_dir=root / "missing",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "closeout",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertTrue(summary["failures"])
            self.assertTrue((root / "closeout" / "summary.json").is_file())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
