from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.sparse_acsr_per_token_churn_fingerprint import (
    REQUIRED_ARTIFACTS,
    run_sparse_acsr_per_token_churn_fingerprint,
)


class SparseACSRPerTokenChurnFingerprintTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_sparse_acsr_per_token_churn_fingerprint(
                common_benchmark_dir=root / "missing_common",
                sparse_gate_dir=root / "missing_gate",
                dense_observables_dir=root / "missing_dense",
                mlp_fingerprint_dir=root / "missing_mlp",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "sparse_acsr_per_token_churn_fingerprint_blocked_by_missing_sparse_fields",
            )
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_current_proxy_sparse_rows_block_on_missing_churn_and_raw_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            gate = root / "gate"
            dense = root / "dense"
            mlp = root / "mlp"
            _write_common(common, include_sparse_churn_raw=False)
            _write_gate(gate)
            _write_dense(dense)
            _write_mlp(mlp)

            summary = run_sparse_acsr_per_token_churn_fingerprint(
                common_benchmark_dir=common,
                sparse_gate_dir=gate,
                dense_observables_dir=dense,
                mlp_fingerprint_dir=mlp,
                out_dir=root / "out",
                min_heldout_rows_per_sparse_arm=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertGreater(summary["sparse_proxy_strata_row_count"], 0)
            self.assertTrue(
                any(
                    row["criterion"] == "sparse_churn_fields_present" and not row["passed"]
                    for row in summary["criteria"]
                )
            )
            with (root / "out" / "missing_field_matrix.csv").open(newline="", encoding="utf-8") as handle:
                matrix = list(csv.DictReader(handle))
            primary = next(row for row in matrix if row["arm"] == "sparse_contextual_topk2")
            self.assertIn("logit_mse_vs_base", primary["missing_churn_fields"])
            self.assertIn("base_logits", primary["missing_raw_fields"])

    def test_passes_when_sparse_churn_and_raw_fields_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            common = root / "common"
            gate = root / "gate"
            dense = root / "dense"
            mlp = root / "mlp"
            _write_common(common, include_sparse_churn_raw=True)
            _write_gate(gate)
            _write_dense(dense)
            _write_mlp(mlp)

            summary = run_sparse_acsr_per_token_churn_fingerprint(
                common_benchmark_dir=common,
                sparse_gate_dir=gate,
                dense_observables_dir=dense,
                mlp_fingerprint_dir=mlp,
                out_dir=root / "out",
                min_heldout_rows_per_sparse_arm=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "sparse_acsr_per_token_churn_fingerprint_available")
            self.assertEqual(
                summary["selected_next_step"],
                "join sparse, dense, and MLP per-token rows into one CE/L2/churn matched intervention decision report",
            )


def _write_common(path: Path, *, include_sparse_churn_raw: bool) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "acsr_common_causal_residual_benchmark_supported"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "arm_metrics.csv",
        [
            {
                "arm": "sparse_contextual_topk2",
                "heldout_ce_loss": 2.9,
                "heldout_delta_vs_base_ce": -0.4,
                "heldout_residual_update_l2": 1.0,
                "active_params_proxy": 192,
            }
        ],
    )
    rows = []
    for arm in ("sparse_contextual_topk2", "sparse_rank_matched_topk1"):
        for index in range(6):
            row = {
                "arm": arm,
                "token_index": index,
                "position_index": index,
                "split": "heldout" if index % 2 else "train",
                "base_ce_loss": 3.5,
                "ce_loss": 3.0 + index * 0.01,
                "delta_vs_base_ce": -0.5 + index * 0.01,
                "residual_update_l2": 1.0,
            }
            if include_sparse_churn_raw:
                row.update(
                    {
                        "logit_mse_vs_base": 0.01,
                        "prediction_changed_vs_base": "False",
                        "residual_update_vector": "[0.1, 0.2]",
                        "base_logits": "[0.1, 0.2]",
                        "candidate_logits": "[0.2, 0.3]",
                    }
                )
            rows.append(row)
    _write_csv(path / "per_token_metrics.csv", rows)


def _write_gate(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "acsr_sparse_dense_mechanism_gate_blocked"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "mechanism_metrics.csv",
        [
            {
                "arm": "acsr_mlp_predicted_future",
                "ce_loss": 2.876,
                "functional_churn": 0.3,
                "anchor_kl_or_logit_mse": 0.01,
                "intervention_fingerprint_purity": 1.0,
            }
        ],
    )


def _write_dense(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for arm in ("dense_rank16_best_norm", "dense_rank24_best_norm", "parameter_matched_causal_mlp_control"):
        for index in range(6):
            rows.append(
                {
                    "arm": arm,
                    "split": "heldout" if index % 2 else "anchor",
                    "ce_loss": 3.2,
                    "delta_vs_base_ce": -0.2,
                    "residual_update_l2": 1.1,
                    "logit_mse_vs_base": 0.02,
                    "prediction_changed_vs_base": "False",
                    "residual_update_vector": "[0.1]",
                    "base_logits": "[0.1, 0.2]",
                    "candidate_logits": "[0.2, 0.3]",
                }
            )
    _write_csv(path / "per_token_observables.csv", rows)


def _write_mlp(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "pass", "decision": "mlp_churn_intervention_fingerprint_scaled_assay_completed"}) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "available_arms.csv",
        [{"arm": "dense_rank24_best_norm", "functional_churn": 0.2}],
    )


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
