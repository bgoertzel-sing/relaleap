from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.core_periphery_pc_column_nonsynthetic_pilot import (
    REQUIRED_ARTIFACTS,
    REQUIRED_VARIANTS,
    _VariantSpec,
    _make_model,
    run_core_periphery_pc_column_nonsynthetic_pilot,
)


class CorePeripheryPCColumnNonSyntheticPilotTest(unittest.TestCase):
    def test_missing_design_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = _write_config(root / "config.yaml")

            summary = run_core_periphery_pc_column_nonsynthetic_pilot(
                design_path=root / "missing.json",
                config_path=config,
                out_dir=root / "out",
                train_steps=1,
                seed=2,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(
                summary["claim_status"],
                "runtime_or_artifact_contract_failed",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(
                any(
                    row["criterion"] == "design_present" and not row["passed"]
                    for row in summary["gate_criteria"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_runs_local_frozen_hidden_state_pilot_with_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design.json"
            design.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "scientific_gate": "ready_for_local_nonsynthetic_pilot_implementation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = _write_config(root / "config.yaml")

            summary = run_core_periphery_pc_column_nonsynthetic_pilot(
                design_path=design,
                config_path=config,
                out_dir=root / "out",
                train_steps=1,
                seed=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertFalse(summary["requires_gpu_now"])
            self.assertEqual(summary["runtime_error"], "")
            self.assertIn(
                summary["scientific_gate"],
                {"blocked", "ready_for_local_repeat_only"},
            )
            variants = {row["variant"] for row in summary["variant_metrics"]}
            self.assertTrue(set(REQUIRED_VARIANTS).issubset(variants))
            self.assertEqual(
                summary["primary_result"]["primary_variant"],
                "contrastive_residual_periphery",
            )
            self.assertGreaterEqual(
                len(summary["intervention_fingerprints"]),
                len(REQUIRED_VARIANTS) * 2,
            )
            manifest_fields = {row["field"] for row in summary["hidden_state_manifest"]}
            self.assertTrue(
                {
                    "frozen_base_hidden_train",
                    "frozen_base_hidden_anchor",
                    "frozen_base_hidden_heldout",
                    "frozen_base_logits_anchor",
                    "teacher_hidden_delta_training_only",
                }.issubset(manifest_fields)
            )
            gate_names = {row["criterion"] for row in summary["gate_criteria"]}
            self.assertIn("required_variants_present", gate_names)
            self.assertIn("matched_dense_retention", gate_names)
            self.assertIn("matched_mlp_retention", gate_names)
            self.assertIn("ce_guardrail_not_worse_than_null", gate_names)
            self.assertIn("periphery_effective_residual_nonzero", gate_names)
            self.assertIn("periphery_heldout_utility_positive", gate_names)
            self.assertIn("core_periphery_update_norm_ratio", summary["primary_result"])
            self.assertIn("periphery_first_prune_delta_heldout", summary["primary_result"])
            self.assertIn("periphery_train_utility_delta", summary["primary_result"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)
            with (root / "out" / "variant_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all("heldout_ce" in row for row in rows))
            self.assertTrue(all("anchor_kl_drift" in row for row in rows))
            self.assertTrue(all("finite_update_commutator" in row for row in rows))
            self.assertTrue(all("core_first_prune_delta_heldout" in row for row in rows))
            self.assertTrue(all("periphery_train_utility_delta" in row for row in rows))
            with (root / "out" / "failed_gate_forensics.csv").open(newline="", encoding="utf-8") as handle:
                forensic_rows = list(csv.DictReader(handle))
            self.assertTrue(
                any(row["variant"] == "contrastive_residual_periphery" for row in forensic_rows)
            )
            self.assertTrue(all("anchor_kl_minus_mlp" in row for row in forensic_rows))

    def test_repaired_arms_use_shared_core_and_complementary_periphery_subspace(self) -> None:
        import torch
        import torch.nn as nn

        for name, training_mode in (
            ("repaired_shared_core_residual_periphery", "shared_core_residual_periphery"),
            ("contrastive_residual_periphery", "contrastive_residual_periphery"),
        ):
            model = _make_model(
                torch,
                nn,
                hidden_dim=8,
                num_columns=4,
                top_k=2,
                spec=_VariantSpec(
                    name,
                    "candidate",
                    training_mode=training_mode,
                ),
            )

            self.assertEqual(model.core_mask.tolist(), [1.0] * 8)
            self.assertEqual(model.periphery_mask.tolist(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


def _write_config(path: Path) -> Path:
    path.write_text(
        """
run:
  experiment_id: test_core_periphery_nonsynthetic
  seed: 2
  max_steps: 2
data:
  dataset: tiny_shakespeare_char
  seq_len: 16
model:
  base:
    layers: 1
    hidden_dim: 16
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    support_router: contextual_mlp
""".lstrip(),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
