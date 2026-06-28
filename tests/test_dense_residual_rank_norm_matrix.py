from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_residual_rank_norm_matrix import (
    REQUIRED_ARTIFACTS,
    run_dense_residual_rank_norm_matrix,
)


class DenseResidualRankNormMatrixTest(unittest.TestCase):
    def test_missing_followup_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = run_dense_residual_rank_norm_matrix(
                config_path=root / "missing.yaml",
                followup_dir=root / "missing_followup",
                out_dir=root / "out",
                dense_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "dense_rank_norm_matrix_failed_closed")
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_tiny_matrix_smoke_runs_and_records_rank_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            followup = root / "followup"
            followup.mkdir()
            _write_tiny_config(config)
            (followup / "summary.json").write_text(
                json.dumps({"status": "pass"}) + "\n",
                encoding="utf-8",
            )
            (followup / "next_matrix.csv").write_text(
                "candidate,rank,norm_scale_vs_sparse_topk2,purpose,status\n"
                + "\n".join(
                    f"dense_causal_rank{rank}_norm_scale_{scale:.2f},{rank},{scale:.2f},unit,selected_for_next_local_cpu_matrix"
                    for rank in (1, 4, 8, 16, 24)
                    for scale in (0.5, 0.75, 1.0)
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_dense_residual_rank_norm_matrix(
                config_path=config,
                followup_dir=followup,
                out_dir=root / "out",
                ranks=(1,),
                norm_scales=(0.5,),
                dense_steps=1,
            )

            self.assertIn(summary["status"], {"pass", "fail"})
            self.assertEqual(summary["matrix_cell_count"], 1)
            self.assertEqual(summary["rank_summary_rows"][0]["rank"], 1)
            self.assertTrue((root / "out" / "matrix_metrics.csv").is_file())
            matrix = (root / "out" / "matrix_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("dense_causal_rank1_norm_scale_0.50", matrix)
            per_token = (root / "out" / "per_token_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("delta_vs_base_ce", per_token)


def _write_tiny_config(path: Path) -> None:
    path.write_text(
        """
run:
  experiment_id: dense_matrix_unit_smoke
  seed: 7

data:
  dataset: tiny_shakespeare_word
  seq_len: 8

model:
  base:
    layers: 1
    hidden_dim: 16
  columns:
    num_columns: 6
    atoms_per_column: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
