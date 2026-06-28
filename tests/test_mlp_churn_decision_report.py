from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mlp_churn_decision_report import (
    REQUIRED_ARTIFACTS,
    run_mlp_churn_decision_report,
)


class MLPChurnDecisionReportTest(unittest.TestCase):
    def test_returns_to_sparse_when_scaled_mlp_not_dense_dominant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fingerprint = root / "fingerprint"
            _write_fingerprint_packet(fingerprint, scaled_mlp_ce=3.82, raw_mlp_churn=0.84)

            summary = run_mlp_churn_decision_report(
                fingerprint_dir=fingerprint,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "return_to_sparse_acsr_support_diagnostics")
            self.assertEqual(
                summary["claim_status"],
                "raw_mlp_high_power_high_churn_scaled_mlp_not_dense_dominant",
            )
            self.assertEqual(
                summary["selected_next_action"],
                "extract_sparse_acsr_per_token_churn_fingerprints",
            )
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(
                any(
                    row["criterion"] == "scaled_mlp_matches_or_beats_dense_at_l2"
                    and not row["passed"]
                    for row in summary["criteria"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_norm_budgeted_mlp_variant_when_scaled_match_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fingerprint = root / "fingerprint"
            _write_fingerprint_packet(fingerprint, scaled_mlp_ce=3.76, raw_mlp_churn=0.25)

            summary = run_mlp_churn_decision_report(
                fingerprint_dir=fingerprint,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "norm_budgeted_churn_regularized_mlp_variant_warranted",
            )
            self.assertEqual(
                summary["selected_next_action"],
                "design_norm_budgeted_churn_regularized_mlp_variant",
            )
            dispositions = {
                row["candidate_action"]: row["disposition"]
                for row in summary["candidate_actions"]
            }
            self.assertEqual(
                dispositions["design_norm_budgeted_churn_regularized_mlp_variant"],
                "selected",
            )

    def test_missing_fingerprint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_mlp_churn_decision_report(
                fingerprint_dir=root / "missing",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "mlp_churn_decision_blocked_missing_fingerprint")
            self.assertEqual(summary["selected_next_action"], "rerun_mlp_churn_intervention_fingerprint")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _write_fingerprint_packet(
    path: Path,
    *,
    scaled_mlp_ce: float,
    raw_mlp_churn: float,
) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "mlp_churn_intervention_fingerprint_scaled_assay_completed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "scaled_interventions.csv",
        [
            _scaled_row("dense_rank24_best_norm", 1.0, 3.767, 1.0, 0.014, 0.24),
            _scaled_row("parameter_matched_causal_mlp_control", 0.25, scaled_mlp_ce, 1.1, 0.010, 0.17),
            _scaled_row("parameter_matched_causal_mlp_control", 1.0, 2.87, 4.4, 0.15, raw_mlp_churn),
        ],
    )
    _write_csv(
        path / "scaled_match_summary.csv",
        [
            {
                "match_type": "residual_l2",
                "reference_arm": "dense_rank24_best_norm",
                "reference_residual_l2": 1.0,
                "reference_ce_loss": 3.767,
                "arm": "parameter_matched_causal_mlp_control",
                "lambda": 0.25,
                "ce_loss": scaled_mlp_ce,
                "residual_update_l2": 1.1,
                "logit_mse_vs_base": 0.010,
                "prediction_changed_vs_base": 0.17,
                "distance": 0.1,
            },
            {
                "match_type": "ce_loss",
                "reference_arm": "dense_rank24_best_norm",
                "reference_residual_l2": 1.0,
                "reference_ce_loss": 3.767,
                "arm": "parameter_matched_causal_mlp_control",
                "lambda": 0.25,
                "ce_loss": scaled_mlp_ce,
                "residual_update_l2": 1.1,
                "logit_mse_vs_base": 0.010,
                "prediction_changed_vs_base": 0.17,
                "distance": abs(scaled_mlp_ce - 3.767),
            },
        ],
    )
    _write_csv(
        path / "available_arms.csv",
        [
            {
                "arm": "acsr_mlp_predicted_future",
                "aggregate_ce_loss": 2.876,
                "aggregate_residual_l2": "",
                "anchor_kl_or_logit_mse": 0.013,
                "functional_churn": 0.30,
            }
        ],
    )


def _scaled_row(
    arm: str,
    lam: float,
    ce_loss: float,
    residual_l2: float,
    logit_mse: float,
    churn: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "lambda": lam,
        "row_count": 128,
        "target_inference_failures": 0,
        "ce_loss": ce_loss,
        "delta_vs_base_ce": -0.1,
        "residual_update_l2": residual_l2,
        "logit_mse_vs_base": logit_mse,
        "prediction_changed_vs_base": churn,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
