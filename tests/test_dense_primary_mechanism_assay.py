from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_primary_mechanism_assay import (
    REQUIRED_ARTIFACTS,
    run_dense_primary_mechanism_assay,
)


class DensePrimaryMechanismAssayTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_primary_mechanism_assay(
                gate_dir=root / "missing_gate",
                stratified_decision_dir=root / "missing_decision",
                dense_observables_dir=root / "missing_observables",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "dense_primary_mechanism_assay_failed_closed")
            self.assertEqual(summary["primary_arm"], "")
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_selects_dense_rank16_when_coverage_and_fields_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "gate"
            decision = root / "decision"
            observables = root / "observables"
            _write_gate(gate)
            _write_stratified_decision(decision)
            _write_per_token_observables(observables)

            summary = run_dense_primary_mechanism_assay(
                gate_dir=gate,
                stratified_decision_dir=decision,
                dense_observables_dir=observables,
                out_dir=root / "out",
                min_per_token_rows_per_arm=4,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "dense_primary_mechanism_assay_selected")
            self.assertEqual(summary["primary_arm"], "dense_rank16_best_norm")
            self.assertFalse(summary["requires_gpu_now"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Primary arm: `dense_rank16_best_norm`", notes)


def _write_gate(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "acsr_sparse_dense_mechanism_gate_blocked"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "mechanism_metrics.csv",
        [
            _metric_row("dense_rank16_best_norm", "dense_rank_control", 2.0, 0.01, 0.2, -0.3, 1.0, 10),
            _metric_row("dense_rank24_best_norm", "dense_rank_control", 2.1, 0.02, 0.3, -0.2, 0.9, 20),
            _metric_row(
                "parameter_matched_causal_mlp_control",
                "parameter_matched_dense_control",
                2.2,
                0.03,
                0.4,
                -0.1,
                0.8,
                30,
            ),
        ],
    )


def _write_stratified_decision(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "demote_acsr_sparse_columns_to_diagnostics"}) + "\n",
        encoding="utf-8",
    )


def _write_per_token_observables(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for arm in ("dense_rank16_best_norm", "dense_rank24_best_norm", "parameter_matched_causal_mlp_control"):
        for index in range(8):
            rows.append(
                {
                    "arm": arm,
                    "source_arm": arm,
                    "rank": "",
                    "token_index": index,
                    "position_index": index,
                    "split": "heldout" if index % 2 else "anchor",
                    "base_ce_loss": 2.0 + index,
                    "ce_loss": 1.8 + index,
                    "delta_vs_base_ce": -0.2,
                    "residual_update_l2": 0.1 * (index + 1),
                    "logit_mse_vs_base": 0.01 * (index + 1),
                    "prediction_changed_vs_base": index % 3 == 0,
                }
            )
    _write_csv(path / "per_token_observables.csv", rows)


def _metric_row(
    arm: str,
    family: str,
    ce_loss: float,
    anchor: float,
    churn: float,
    retention: float,
    purity: float,
    params: int,
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "ce_loss": ce_loss,
        "oracle_regret": "",
        "residual_l2": 1.0,
        "active_rank_or_topk": 16,
        "active_params": params,
        "anchor_kl_or_logit_mse": anchor,
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
