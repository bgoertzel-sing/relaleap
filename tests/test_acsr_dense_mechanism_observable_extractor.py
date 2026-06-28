from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_dense_mechanism_observable_extractor import (
    REQUIRED_ARTIFACTS,
    run_acsr_dense_mechanism_observable_extractor,
)


class ACSRDenseMechanismObservableExtractorTest(unittest.TestCase):
    def test_missing_dense_matrix_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text("run: {}\n", encoding="utf-8")

            summary = run_acsr_dense_mechanism_observable_extractor(
                config_path=config,
                dense_matrix_dir=root / "missing_dense",
                out_dir=root / "out",
                dense_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_dense_mechanism_observables_failed_closed")
            self.assertFalse(any(row["passed"] for row in summary["gate_criteria"] if row["criterion"] == "dense_matrix_passed"))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_preflight_requires_rank16_and_rank24_best_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.yaml"
            config.write_text("run: {}\n", encoding="utf-8")
            dense = root / "dense"
            dense.mkdir()
            (dense / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "sparse_reference": {
                            "heldout_residual_update_l2": 1.0,
                            "active_params_proxy": 192,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (dense / "rank_summary.csv").write_text(
                "rank,best_arm\n16,dense_causal_rank16_norm_scale_1.00\n24,dense_causal_rank24_norm_scale_1.00\n",
                encoding="utf-8",
            )
            (dense / "matrix_metrics.csv").write_text(
                "arm,rank,target_heldout_l2,target_active_params_proxy\n"
                "dense_causal_rank16_norm_scale_1.00,16,1.0,192\n",
                encoding="utf-8",
            )

            summary = run_acsr_dense_mechanism_observable_extractor(
                config_path=config,
                dense_matrix_dir=dense,
                out_dir=root / "out",
                dense_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(
                any(
                    row["criterion"] == "rank16_24_best_arms_present"
                    and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )


if __name__ == "__main__":
    unittest.main()
