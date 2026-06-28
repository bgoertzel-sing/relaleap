from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.mlp_dense_heldout_mechanism_followup import (
    REQUIRED_ARTIFACTS,
    run_mlp_dense_heldout_mechanism_followup,
)


class MLPDenseHeldoutMechanismFollowupTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_mlp_dense_heldout_mechanism_followup(
                primary_assay_dir=root / "missing_primary",
                dense_observables_dir=root / "missing_observables",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "mlp_dense_heldout_mechanism_followup_failed_closed")
            self.assertEqual(summary["primary_arm"], "")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_passes_with_mlp_churn_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "primary"
            observables = root / "observables"
            _write_primary_assay(primary)
            _write_observables(observables)

            summary = run_mlp_dense_heldout_mechanism_followup(
                primary_assay_dir=primary,
                dense_observables_dir=observables,
                out_dir=root / "out",
                min_heldout_rows_per_arm=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "mlp_primary_with_functional_churn_tradeoff")
            self.assertEqual(summary["primary_arm"], "parameter_matched_causal_mlp_control")
            self.assertFalse(summary["requires_gpu_now"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("functional_churn_tradeoff", notes)


def _write_primary_assay(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "dense_primary_mechanism_assay_selected",
                "primary_arm": "parameter_matched_causal_mlp_control",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "candidate_scorecard.csv",
        [
            _scorecard_row("dense_rank16_best_norm", 0.2, -0.3, 0.9, 10),
            _scorecard_row("dense_rank24_best_norm", 0.3, -0.4, 0.9, 20),
            _scorecard_row("parameter_matched_causal_mlp_control", 0.8, -1.2, 1.0, 30),
        ],
    )


def _write_observables(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    ce_by_arm = {
        "dense_rank16_best_norm": 3.0,
        "dense_rank24_best_norm": 2.8,
        "parameter_matched_causal_mlp_control": 2.0,
    }
    for arm, ce_loss in ce_by_arm.items():
        for index in range(8):
            rows.append(
                {
                    "arm": arm,
                    "source_arm": arm,
                    "rank": "",
                    "token_index": index,
                    "position_index": index,
                    "split": "heldout" if index % 2 else "anchor",
                    "base_ce_loss": 3.5,
                    "ce_loss": ce_loss,
                    "delta_vs_base_ce": ce_loss - 3.5,
                    "residual_update_l2": 0.1 * (index + 1),
                    "logit_mse_vs_base": 0.01 * (index + 1),
                    "prediction_changed_vs_base": arm == "parameter_matched_causal_mlp_control",
                }
            )
    _write_csv(path / "per_token_observables.csv", rows)


def _scorecard_row(
    arm: str,
    churn: float,
    retention: float,
    purity: float,
    params: int,
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": "parameter_matched_dense_control" if "mlp" in arm else "dense_rank_control",
        "ce_loss": 1.0,
        "residual_l2": 1.0,
        "active_rank_or_topk": 2,
        "active_params": params,
        "anchor_kl_or_logit_mse": 0.1,
        "functional_churn": churn,
        "retention_or_forgetting": retention,
        "intervention_fingerprint_purity": purity,
        "mechanism_fields_present": True,
        "missing_mechanism_fields": "",
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
