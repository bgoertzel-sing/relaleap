from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_mechanism_stratified_decision import (
    REQUIRED_ARTIFACTS,
    run_acsr_mechanism_stratified_decision,
)


class ACSRMechanismStratifiedDecisionTest(unittest.TestCase):
    def test_missing_gate_fails_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_acsr_mechanism_stratified_decision(
                gate_dir=root / "missing_gate",
                dense_observables_dir=root / "missing_observables",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "demote_acsr_sparse_columns_to_diagnostics")
            self.assertFalse(summary["continue_sparse_columns"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_blocked_gate_demotes_when_sparse_strata_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "gate"
            observables = root / "observables"
            _write_gate(gate, decision="acsr_sparse_dense_mechanism_gate_blocked")
            _write_per_token_observables(observables)

            summary = run_acsr_mechanism_stratified_decision(
                gate_dir=gate,
                dense_observables_dir=observables,
                out_dir=root / "out",
                min_strata_per_control=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "demote_acsr_sparse_columns_to_diagnostics")
            self.assertFalse(summary["continue_sparse_columns"])
            self.assertTrue(
                any(
                    row["criterion"] == "sparse_per_token_strata_available"
                    and not row["passed"]
                    for row in summary["criteria"]
                )
            )
            self.assertGreater(summary["per_token_strata_row_count"], 0)
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Continue sparse columns: `False`", notes)


def _write_gate(path: Path, *, decision: str) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": decision, "claim_status": "blocked"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "mechanism_metrics.csv",
        [
            _metric_row("acsr_mlp_predicted_future", "sparse_support", 2.8, 0.01, 0.3, -0.1, 1.0),
            _metric_row("dense_rank16_best_norm", "dense_rank_control", 2.7, 0.02, 0.2, -0.2, 1.0),
            _metric_row("dense_rank24_best_norm", "dense_rank_control", 2.6, 0.03, 0.4, -0.3, 1.0),
            _metric_row("parameter_matched_causal_mlp_control", "parameter_matched_dense_control", 2.79, 0.2, 0.7, -1.0, 1.0),
        ],
    )


def _write_per_token_observables(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for arm in ("dense_rank16_best_norm", "dense_rank24_best_norm", "parameter_matched_causal_mlp_control"):
        for index in range(6):
            rows.append(
                {
                    "arm": arm,
                    "source_arm": arm,
                    "rank": "",
                    "token_index": index,
                    "position_index": index,
                    "split": "heldout" if index % 3 == 0 else "anchor",
                    "base_ce_loss": 2.0 + index,
                    "ce_loss": 1.9 + index,
                    "delta_vs_base_ce": -0.1,
                    "residual_update_l2": 0.1 * (index + 1),
                    "logit_mse_vs_base": 0.01 * (index + 1),
                    "prediction_changed_vs_base": index % 2 == 0,
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
) -> dict[str, object]:
    return {
        "arm": arm,
        "family": family,
        "ce_loss": ce_loss,
        "oracle_regret": "",
        "residual_l2": "",
        "active_rank_or_topk": "",
        "active_params": "",
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
