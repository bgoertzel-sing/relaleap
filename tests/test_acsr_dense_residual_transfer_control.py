from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_dense_residual_transfer_control import (
    REQUIRED_ARTIFACTS,
    _dense_gate_rows,
    _source_probe_metrics,
    run_acsr_dense_residual_transfer_control,
)


class ACSRDenseResidualTransferControlTest(unittest.TestCase):
    def test_missing_source_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_acsr_dense_residual_transfer_control(
                source_probe_dir=root / "missing",
                config_path=root / "missing.yaml",
                out_dir=root / "out",
                dense_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_dense_residual_transfer_control_failed_closed")
            self.assertEqual(summary["claim_status"], "dense_control_not_run")
            self.assertTrue(
                any(row["criterion"] == "source_probe_present" for row in summary["failures"])
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_source_metrics_and_dense_gate_pass_when_sparse_beats_dense(self) -> None:
        rows = []
        for value_path in ("partner_values", "own_values"):
            for arm in ("direct_causal_mlp_baseline", "transfer_objective_router"):
                rows.extend(_per_token_rows(value_path, arm))

        source_metrics = _source_probe_metrics(rows)
        dense_rows = [
            {
                "control": "rank_matched_causal_dense_residual",
                "heldout_delta_vs_base_ce": -0.01,
                "residual_update_l2_mean": 0.3,
            },
            {
                "control": "rank_matched_token_position_dense_residual",
                "heldout_delta_vs_base_ce": 0.02,
                "residual_update_l2_mean": 0.3,
            },
        ]
        gate_rows = _dense_gate_rows(source_metrics, dense_rows)

        self.assertEqual(len(source_metrics), 4)
        self.assertTrue(all(row["passed"] for row in gate_rows), gate_rows)

    def test_dense_gate_fails_when_dense_matches_sparse(self) -> None:
        rows = []
        for value_path in ("partner_values", "own_values"):
            for arm in ("direct_causal_mlp_baseline", "transfer_objective_router"):
                rows.extend(_per_token_rows(value_path, arm))

        source_metrics = _source_probe_metrics(rows)
        dense_rows = [
            {
                "control": "rank_matched_causal_dense_residual",
                "heldout_delta_vs_base_ce": -0.08,
                "residual_update_l2_mean": 0.3,
            },
            {
                "control": "rank_matched_token_position_dense_residual",
                "heldout_delta_vs_base_ce": 0.02,
                "residual_update_l2_mean": 0.3,
            },
        ]
        gate_rows = _dense_gate_rows(source_metrics, dense_rows)

        self.assertTrue(
            any(
                row["criterion"] == "sparse_transfer_beats_causal_dense_control"
                and not row["passed"]
                for row in gate_rows
            )
        )


def _per_token_rows(value_path: str, arm: str) -> list[dict[str, object]]:
    rows = []
    for index in range(32):
        position = index % 8
        heldout = position >= 4
        base_loss = 5.0 + (0.01 * position)
        if value_path == "partner_values":
            adjustment = {
                "direct_causal_mlp_baseline": 0.0,
                "transfer_objective_router": -0.04 if heldout else -0.01,
            }[arm]
        else:
            adjustment = {
                "direct_causal_mlp_baseline": 0.0,
                "transfer_objective_router": 0.01,
            }[arm]
        rows.append(
            {
                "value_path": value_path,
                "arm": arm,
                "token_index": index,
                "ce_loss": base_loss + adjustment,
                "residual_update_l2": 0.2,
                "support": "1;2",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
