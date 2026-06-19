from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.run import run
from relaleap.smoke import (
    ResidualColumns,
    _hep_support_instability,
    run_phase0_smoke,
)


CONFIG = {
    "run": {"experiment_id": "test_smoke", "seed": 1, "max_steps": 2},
    "data": {"dataset": "tiny_shakespeare_char", "seq_len": 32},
    "model": {
        "base": {"layers": 2, "hidden_dim": 32},
        "columns": {
            "num_columns": 8,
            "atoms_per_column": 4,
            "top_k": 1,
            "insertion_sites": 1,
        },
    },
    "inference": {"pc_steps": 1, "hep_alpha": 0.0},
    "outputs": {
        "require_summary_json": True,
        "require_metrics_csv": True,
        "require_notes_md": True,
    },
}


class Phase0SmokeTest(unittest.TestCase):
    def test_phase0_invariants_pass(self) -> None:
        try:
            result = run_phase0_smoke(CONFIG)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.invariants["zero_init_identity"])
        self.assertTrue(result.invariants["frozen_base_unchanged"])
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertTrue(result.invariants["residual_parameters_updated"])
        self.assertAlmostEqual(result.base_loss, result.zero_init_loss)
        self.assertAlmostEqual(result.initial_loss, result.zero_init_loss)
        self.assertEqual(result.training_steps, CONFIG["run"]["max_steps"])
        self.assertEqual(result.residual_objective, "supervised_ce")
        self.assertEqual(len(result.to_metric_rows()), CONFIG["run"]["max_steps"] + 1)
        self.assertEqual(result.to_metric_rows()[0]["phase"], "initial")
        self.assertTrue(
            all(
                row["residual_objective"] == "supervised_ce"
                for row in result.to_metric_rows()
            )
        )
        self.assertTrue(
            all(row["phase"] == "residual_update" for row in result.to_metric_rows()[1:])
        )
        self.assertEqual(result.to_metric_rows()[-1]["residual_loss"], result.post_step_loss)

    def test_pc_logit_mse_objective_toggle(self) -> None:
        pc_config = copy.deepcopy(CONFIG)
        pc_config["training"] = {"residual_objective": "pc_logit_mse"}

        try:
            result = run_phase0_smoke(pc_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertEqual(result.residual_objective, "pc_logit_mse")
        self.assertTrue(all(result.invariants.values()))
        self.assertEqual(result.to_metric_rows()[0]["phase"], "initial")
        self.assertTrue(
            all(
                row["phase"] == "pc_residual_update"
                for row in result.to_metric_rows()[1:]
            )
        )
        self.assertTrue(
            all(
                row["residual_objective"] == "pc_logit_mse"
                for row in result.to_metric_rows()
            )
        )
        self.assertGreater(result.residual_parameter_delta, 0.0)

    def test_hep_alpha_sweep_records_nonzero_alpha_rows(self) -> None:
        hep_config = copy.deepcopy(CONFIG)
        hep_config["run"]["max_steps"] = 1
        hep_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,0.5,1.0",
        }

        try:
            result = run_phase0_smoke(hep_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertEqual(
            [entry["alpha"] for entry in result.hep_alpha_sweep],
            [0.0, 0.5, 1.0],
        )
        self.assertLessEqual(
            result.hep_alpha_sweep[0]["max_logit_delta_from_ordinary"],
            1e-6,
        )
        sweep_rows = [
            row for row in result.to_metric_rows() if row["phase"] == "hep_sweep"
        ]
        self.assertEqual([row["hep_alpha"] for row in sweep_rows], [0.0, 0.5, 1.0])
        self.assertTrue(
            all(row["hep_loss"] != "" for row in sweep_rows)
        )

    def test_residual_columns_can_pin_selected_support(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        residual = ResidualColumns(
            hidden_dim=2,
            num_columns=3,
            atoms_per_column=1,
            top_k=1,
        )
        with torch.no_grad():
            residual.column_scores.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                )
            )
            residual.atom_values.copy_(
                torch.tensor(
                    [
                        [[-2.0, 5.0]],
                        [[0.0, 10.0]],
                        [[1.0, 1.0]],
                    ]
                )
            )

        hidden = torch.tensor([[[1.0, 0.0]]])
        settled, support = residual(hidden, return_support=True)
        repicked = residual(settled)
        pinned = residual(settled, support_indices=support)

        self.assertEqual(support.tolist(), [[[0]]])
        self.assertNotEqual(repicked.tolist(), pinned.tolist())
        self.assertEqual(pinned.tolist(), [[[-3.0, 10.0]]])

    def test_support_instability_detects_repicked_settling(self) -> None:
        try:
            import torch
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                self.skipTest(str(exc))
            raise

        class IdentityBase:
            def encode(self, inputs):
                return inputs

            def decode(self, hidden):
                return hidden

        residual = ResidualColumns(
            hidden_dim=2,
            num_columns=3,
            atoms_per_column=1,
            top_k=1,
        )
        with torch.no_grad():
            residual.column_scores.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                )
            )
            residual.atom_values.copy_(
                torch.tensor(
                    [
                        [[-2.0, 5.0]],
                        [[0.0, 10.0]],
                        [[1.0, 1.0]],
                    ]
                )
            )

        diagnostic = _hep_support_instability(
            IdentityBase(),
            residual,
            torch.tensor([[[1.0, 0.0]]]),
            pc_steps=2,
            hep_alpha=1.0,
        )

        self.assertTrue(diagnostic["support_changed"])
        self.assertEqual(diagnostic["support_changed_positions"], 1)
        self.assertEqual(diagnostic["support_transition_count"], 1)
        self.assertEqual(diagnostic["support_change_fraction"], 1.0)
        self.assertGreater(diagnostic["pinned_vs_repicked_logit_delta"], 0.0)

    def test_pinned_support_config_is_reported(self) -> None:
        pinned_config = copy.deepcopy(CONFIG)
        pinned_config["run"]["max_steps"] = 1
        pinned_config["model"]["columns"]["pinned_support"] = True
        pinned_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,0.5",
        }

        try:
            result = run_phase0_smoke(pinned_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.pinned_support)
        self.assertTrue(result.to_summary()["pinned_support"])
        self.assertIn("support_change_fraction", result.support_instability)
        self.assertIn("support_instability", result.to_summary())
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])

    def test_support_stress_config_records_nonzero_repicking(self) -> None:
        stress_config = copy.deepcopy(CONFIG)
        stress_config["run"]["max_steps"] = 1
        stress_config["run"]["experiment_id"] = "test_support_stress"
        stress_config["model"]["columns"]["support_stress"] = True
        stress_config["inference"] = {
            "pc_steps": 2,
            "hep_alpha": 0.0,
            "hep_alpha_sweep": "0.0,1.0",
        }

        try:
            result = run_phase0_smoke(stress_config)
        except RuntimeError as exc:
            if "torch" in str(exc):
                self.skipTest(str(exc))
            raise

        self.assertTrue(result.support_stress)
        self.assertTrue(result.invariants["zero_init_identity"])
        self.assertTrue(result.invariants["frozen_base_unchanged"])
        self.assertTrue(result.invariants["hep_alpha_0_equivalence"])
        self.assertGreater(result.support_instability["support_change_fraction"], 0.0)
        self.assertGreater(
            result.support_instability["pinned_vs_repicked_logit_delta"],
            0.0,
        )
        self.assertIn("support_stress", [row["phase"] for row in result.to_metric_rows()])
        alpha1 = [
            entry for entry in result.hep_alpha_sweep if entry["alpha"] == 1.0
        ][0]
        self.assertGreater(alpha1["support_change_fraction"], 0.0)
        self.assertGreater(alpha1["pinned_vs_repicked_logit_delta"], 0.0)

    def test_runner_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  experiment_id: test_smoke",
                        "  seed: 1",
                        "  max_steps: 2",
                        "data:",
                        "  dataset: tiny_shakespeare_char",
                        "  seq_len: 32",
                        "training:",
                        "  residual_objective: supervised_ce",
                        "model:",
                        "  base:",
                        "    layers: 2",
                        "    hidden_dim: 32",
                        "  columns:",
                        "    num_columns: 8",
                        "    atoms_per_column: 4",
                        "    top_k: 1",
                        "    insertion_sites: 1",
                        "inference:",
                        "  pc_steps: 1",
                        "  hep_alpha: 0.0",
                        "outputs:",
                        "  require_summary_json: true",
                        "  require_metrics_csv: true",
                        "  require_notes_md: true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run(config_path, tmp_path / "out")
            if summary["error"] and "torch" in summary["error"]:
                self.skipTest(summary["error"])

            self.assertEqual(summary["status"], "ok")
            self.assertTrue((tmp_path / "out" / "summary.json").is_file())
            self.assertTrue((tmp_path / "out" / "metrics.csv").is_file())
            self.assertTrue((tmp_path / "out" / "notes.md").is_file())
            saved = json.loads((tmp_path / "out" / "summary.json").read_text())
            self.assertTrue(saved["artifact_invariants"]["summary_json"])
            self.assertTrue(saved["phase0"]["invariants"]["zero_init_identity"])
            self.assertIn("base_loss", saved["phase0"])
            self.assertIn("post_step_loss", saved["phase0"])
            self.assertEqual(saved["phase0"]["residual_objective"], "supervised_ce")
            self.assertEqual(saved["phase0"]["training_steps"], 2)

            with (tmp_path / "out" / "metrics.csv").open(newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["phase"] for row in metric_rows],
                ["initial", "residual_update", "residual_update"],
            )
            self.assertEqual([int(row["step"]) for row in metric_rows], [0, 1, 2])
            self.assertNotIn("smoke_loss", metric_rows[0])
            self.assertIn("base_loss", metric_rows[0])
            self.assertIn("residual_loss", metric_rows[0])
            self.assertIn("residual_objective", metric_rows[0])
            self.assertIn("hep_alpha", metric_rows[0])
            self.assertIn("hep_loss", metric_rows[0])
            self.assertIn("hep_support_change_fraction", metric_rows[0])
            self.assertIn("hep_pinned_vs_repicked_logit_delta", metric_rows[0])
            self.assertEqual(metric_rows[0]["residual_objective"], "supervised_ce")
            self.assertEqual(
                float(metric_rows[-1]["residual_loss"]),
                saved["final_smoke_loss"],
            )


if __name__ == "__main__":
    unittest.main()
