from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_causal_adequacy_matrix import (
    INSUFFICIENT_EVIDENCE,
    PREDICTIVE_DEFAULT_CAUSAL_ADEQUACY_NOT_ESTABLISHED,
    run_promoted_topk2_causal_adequacy_matrix,
)


class PromotedTopk2CausalAdequacyMatrixTest(unittest.TestCase):
    def test_builds_matrix_and_blocks_causal_adequacy_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = _write_source_packets(root)

            summary = run_promoted_topk2_causal_adequacy_matrix(
                retention_synthesis_dir=dirs["retention"],
                finite_update_matrix_dir=dirs["finite"],
                functional_churn_dir=dirs["functional"],
                support_selection_dir=dirs["support"],
                deconfounded_dir=dirs["deconfounded"],
                topk1_causal_retention_dir=dirs["topk1"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                PREDICTIVE_DEFAULT_CAUSAL_ADEQUACY_NOT_ESTABLISHED,
            )
            self.assertEqual(len(summary["matrix_rows"]), 4)
            self.assertTrue(summary["signals"]["ce_guardrail_passed"])
            self.assertTrue(summary["signals"]["oracle_support_regret_low"])
            self.assertTrue(summary["signals"]["topk2_predictive_control_win"])
            self.assertTrue(summary["signals"]["topk1_cleaner_retention_control"])
            self.assertTrue(summary["signals"]["finite_update_risk_high_vs_topk1"])
            self.assertFalse(summary["signals"]["intervention_cleanliness_gate_passed"])
            self.assertFalse(summary["signals"]["promoted_topk2_causal_adequacy_supported"])
            self.assertGreater(
                summary["metrics"]["topk2_to_topk1_finite_update_logit_mse_ratio"],
                20.0,
            )
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "causal_adequacy_matrix.csv").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_fails_closed_when_required_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = _write_source_packets(root)

            summary = run_promoted_topk2_causal_adequacy_matrix(
                retention_synthesis_dir=dirs["retention"],
                finite_update_matrix_dir=root / "missing_finite",
                functional_churn_dir=dirs["functional"],
                support_selection_dir=dirs["support"],
                deconfounded_dir=dirs["deconfounded"],
                topk1_causal_retention_dir=dirs["topk1"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertTrue(
                any(
                    failure["source"] == "finite_update_control_matrix"
                    and failure["field"] == "summary_json"
                    for failure in summary["failures"]
                )
            )


def _write_source_packets(root: Path) -> dict[str, Path]:
    dirs = {
        "retention": root / "retention",
        "finite": root / "finite",
        "functional": root / "functional",
        "support": root / "support",
        "deconfounded": root / "deconfounded",
        "topk1": root / "topk1",
    }
    _write_json(
        dirs["retention"] / "summary.json",
        {
            "status": "pass",
            "decision": "contextual_topk2_router_default_topk1_diagnostic",
            "metrics": {
                "mean_topk2_transfer_ce_improvement": 0.93,
                "mean_topk1_transfer_ce_improvement": 0.96,
                "mean_random_fixed_topk2_transfer_ce_improvement": 0.65,
                "mean_dense_transfer_ce_improvement": 0.48,
                "mean_topk2_support_churn_after_transfer": 0.86,
                "mean_topk1_support_churn_after_transfer": 0.006,
            },
        },
    )
    _write_json(
        dirs["finite"] / "summary.json",
        {
            "status": "pass",
            "decision": "finite_update_control_matrix_ready",
            "metrics": {
                "topk2_mean_logit_mse": 0.24,
                "topk1_mean_logit_mse": 0.009,
                "random_fixed_topk2_mean_logit_mse": 0.35,
                "dense_active_rank_mean_logit_mse": 0.064,
                "topk2_mean_ce_abs_delta": 0.40,
                "topk1_mean_ce_abs_delta": 0.10,
                "random_fixed_topk2_mean_ce_abs_delta": 0.44,
                "dense_active_rank_mean_ce_abs_delta": 0.066,
                "topk2_mean_residual_delta_l2": 5.1,
                "topk1_mean_residual_delta_l2": 1.3,
            },
        },
    )
    _write_json(
        dirs["functional"] / "summary.json",
        {
            "status": "pass",
            "decision": "support_identity_churn_functional_impact_bounded_with_commutator_risk",
        },
    )
    _write_json(
        dirs["support"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_support_selection_quality_established",
            "metrics": {
                "oracle_support_regret": 0.0025,
                "oracle_support_regret_positive_fraction": 0.055,
            },
        },
    )
    _write_json(
        dirs["deconfounded"] / "summary.json",
        {
            "status": "pass",
            "decision": "topk2_comparative_causal_cooperation_not_supported",
            "evidence": {
                "topk2_alpha0_ce_loss": 2.912,
                "topk1_alpha0_ce_loss": 2.866,
                "topk2_ce_deficit_vs_topk1": 0.046,
                "topk2_fixed_support_cleaner_strata_fraction": 0.65,
                "topk2_functional_churn_cleaner_strata_fraction": 0.61,
                "topk2_incremental_pair_gain_positive_strata_fraction": 0.65,
            },
        },
    )
    _write_json(
        dirs["topk1"] / "summary.json",
        {
            "status": "pass",
            "decision": "causal_retention_claim_blocked_by_deployable_gate",
            "evidence": {
                "metrics": {
                    "selected_singleton_gain_mean": 1.0,
                    "deployable_gain_minus_ungated": -0.043,
                    "deployable_offcontext_harm_suppression_fraction": 0.129,
                }
            },
        },
    )
    return dirs


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
