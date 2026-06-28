from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mlp_churn_intervention_fingerprint import (
    REQUIRED_ARTIFACTS,
    run_mlp_churn_intervention_fingerprint,
)


class MLPChurnInterventionFingerprintTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_mlp_churn_intervention_fingerprint(
                followup_dir=root / "missing_followup",
                dense_observables_dir=root / "missing_observables",
                sparse_gate_dir=root / "missing_sparse",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "mlp_churn_intervention_fingerprint_blocked_by_missing_raw_intervention_fields",
            )
            self.assertIn("residual_update_vector", summary["missing_required_fields"]["raw_intervention_fields"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_proxy_report_lists_current_raw_field_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            followup = root / "followup"
            observables = root / "observables"
            sparse = root / "sparse"
            _write_followup(followup)
            _write_observables(observables)
            _write_sparse_gate(sparse)

            summary = run_mlp_churn_intervention_fingerprint(
                followup_dir=followup,
                dense_observables_dir=observables,
                sparse_gate_dir=sparse,
                out_dir=root / "out",
                min_heldout_rows_per_arm=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["claim_status"],
                "proxy_fingerprint_written_but_raw_intervention_assay_not_decisive",
            )
            self.assertGreater(summary["matched_curve_row_count"], 0)
            self.assertGreater(summary["fingerprint_strata_row_count"], 0)
            self.assertTrue(
                any(
                    row["criterion"] == "raw_intervention_fields_present"
                    and not row["passed"]
                    for row in summary["criteria"]
                )
            )
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("raw residual update vectors and logits", notes)


def _write_followup(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "primary_arm": "parameter_matched_causal_mlp_control",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "mechanism_comparison.csv",
        [
            _aggregate_row("dense_rank16_best_norm", "dense_rank_control", 3.0, 1.0, 0.2),
            _aggregate_row("dense_rank24_best_norm", "dense_rank_control", 2.9, 1.0, 0.3),
            _aggregate_row("parameter_matched_causal_mlp_control", "parameter_matched_dense_control", 2.0, 4.0, 0.8),
        ],
    )


def _write_observables(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    arms = {
        "dense_rank16_best_norm": 3.0,
        "dense_rank24_best_norm": 2.8,
        "parameter_matched_causal_mlp_control": 2.0,
    }
    for arm, ce_loss in arms.items():
        for index in range(6):
            split = "heldout" if index % 2 else "anchor"
            rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "token_index": index,
                    "position_index": index,
                    "base_ce_loss": 3.5,
                    "ce_loss": ce_loss + index * 0.01,
                    "delta_vs_base_ce": ce_loss + index * 0.01 - 3.5,
                    "residual_update_l2": 0.25 * (index + 1),
                    "logit_mse_vs_base": 0.01 * (index + 1),
                    "prediction_changed_vs_base": arm == "parameter_matched_causal_mlp_control",
                }
            )
    _write_csv(path / "per_token_observables.csv", rows)


def _write_sparse_gate(path: Path) -> None:
    path.mkdir(parents=True)
    _write_csv(
        path / "mechanism_metrics.csv",
        [
            _aggregate_row("acsr_mlp_predicted_future", "sparse_support", 2.1, 1.1, 0.3),
            _aggregate_row("shuffled_predicted_features", "null_support", 3.4, 1.1, 0.7),
        ],
    )


def _aggregate_row(
    arm: str,
    family: str,
    ce_loss: float,
    residual_l2: float,
    churn: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "ce_loss": ce_loss,
        "residual_l2": residual_l2,
        "anchor_kl_or_logit_mse": 0.1,
        "functional_churn": churn,
        "retention_or_forgetting": -0.1,
        "intervention_fingerprint_purity": 1.0,
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
