from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from relaleap.experiments.acsr_dense_mechanism_observable_extractor import (
    REQUIRED_ARTIFACTS,
    _observable_gate_rows,
    _per_token_observable_rows,
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

    def test_per_token_rows_include_raw_intervention_fields(self) -> None:
        base_logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.5, 0.25]]])
        candidate_logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.4, 0.35]]])
        base_losses = torch.tensor([0.3, 0.4])
        losses = torch.tensor([[0.2, 0.6]])
        residual_l2 = torch.tensor([0.5, 0.7])
        residual_update = torch.tensor([[[0.1, 0.2, 0.3], [0.0, -0.1, 0.2], [0.4, 0.0, 0.1]]])
        heldout_mask = torch.tensor([False, True])

        rows = _per_token_observable_rows(
            torch=torch,
            arm="dense_rank16_best_norm",
            rank=16,
            source_arm="dense_causal_rank16_norm_scale_1.00",
            base_logits=base_logits,
            dense_logits=candidate_logits,
            base_losses=base_losses,
            losses=losses,
            l2=residual_l2,
            residual_update=residual_update,
            heldout_mask=heldout_mask,
            seq_len_minus_one=2,
        )

        self.assertEqual(len(rows), 2)
        for field in ("residual_update_vector", "base_logits", "candidate_logits"):
            self.assertIn(field, rows[0])
            self.assertIsInstance(json.loads(rows[0][field]), list)
        criteria = _observable_gate_rows(
            observable_rows=[
                {
                    "arm": "dense_rank16_best_norm",
                    "rank": 16,
                    "anchor_kl_or_logit_mse": 0.1,
                    "functional_churn": 0.0,
                    "retention_or_forgetting": 0.0,
                    "intervention_fingerprint_purity": 1.0,
                },
                {
                    "arm": "dense_rank24_best_norm",
                    "rank": 24,
                    "anchor_kl_or_logit_mse": 0.1,
                    "functional_churn": 0.0,
                    "retention_or_forgetting": 0.0,
                    "intervention_fingerprint_purity": 1.0,
                },
            ],
            control_rows=[
                {
                    "arm": "parameter_matched_causal_mlp_control",
                    "anchor_kl_or_logit_mse": 0.1,
                    "functional_churn": 0.0,
                    "retention_or_forgetting": 0.0,
                    "intervention_fingerprint_purity": 1.0,
                }
            ],
            per_token_rows=rows,
        )
        self.assertTrue(
            any(
                row["criterion"] == "per_token_raw_intervention_fields_present"
                and row["passed"]
                for row in criteria
            )
        )


if __name__ == "__main__":
    unittest.main()
