from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.run import run
from relaleap.smoke import run_phase0_smoke


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
        self.assertEqual(len(result.to_metric_rows()), CONFIG["run"]["max_steps"] + 1)
        self.assertEqual(result.to_metric_rows()[0]["phase"], "initial")
        self.assertTrue(
            all(row["phase"] == "residual_update" for row in result.to_metric_rows()[1:])
        )
        self.assertEqual(result.to_metric_rows()[-1]["residual_loss"], result.post_step_loss)

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
            self.assertEqual(
                float(metric_rows[-1]["residual_loss"]),
                saved["final_smoke_loss"],
            )


if __name__ == "__main__":
    unittest.main()
