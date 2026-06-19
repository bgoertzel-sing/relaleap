from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.compare import (
    DEFAULT_CONFIGS,
    _comparison_entry,
    run_comparison,
)


class ComparisonReportTest(unittest.TestCase):
    def test_comparison_entry_summarizes_loss_trajectory(self) -> None:
        summary = {
            "experiment_id": "char_smoke",
            "status": "ok",
            "error": None,
            "phase0": {
                "residual_objective": "supervised_ce",
                "training_steps": 2,
                "base_loss": 3.0,
                "zero_init_loss": 3.0,
                "invariants": {"zero_init_identity": True},
            },
        }
        rows = [
            {"step": "0", "residual_loss": "3.00000000"},
            {"step": "1", "residual_loss": "2.50000000"},
            {"step": "2", "residual_loss": "2.25000000"},
        ]

        entry = _comparison_entry(
            Path("configs/char_smoke.yaml"),
            Path("out/runs/char_smoke"),
            summary,
            rows,
        )

        self.assertEqual(entry["residual_objective"], "supervised_ce")
        self.assertEqual(entry["initial_residual_loss"], 3.0)
        self.assertEqual(entry["final_residual_loss"], 2.25)
        self.assertEqual(entry["residual_loss_delta"], -0.75)
        self.assertEqual(entry["residual_loss_ratio"], 0.75)
        self.assertEqual(entry["hep_alpha_sweep"], [])

    def test_default_configs_include_supervised_pc_and_hep(self) -> None:
        self.assertEqual(
            [config.name for config in DEFAULT_CONFIGS],
            ["char_smoke.yaml", "char_smoke_pc.yaml", "char_smoke_hep.yaml"],
        )

    def test_run_comparison_writes_top_level_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            configs = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
            for config in configs:
                config.write_text("run:\n  experiment_id: fake\n", encoding="utf-8")

            def fake_run(config_path: Path, run_dir: Path) -> dict[str, object]:
                run_dir.mkdir(parents=True, exist_ok=True)
                experiment_id = config_path.stem
                objective = "supervised_ce" if experiment_id == "a" else "pc_logit_mse"
                with (run_dir / "metrics.csv").open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "step",
                            "phase",
                            "base_loss",
                            "residual_loss",
                            "hep_alpha",
                            "hep_loss",
                            "max_hep_logit_delta_from_ordinary",
                            "status",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {
                                "step": 0,
                                "phase": "initial",
                                "base_loss": "1.00000000",
                                "residual_loss": "1.00000000",
                                "hep_alpha": "",
                                "hep_loss": "",
                                "max_hep_logit_delta_from_ordinary": "",
                                "status": "ok",
                            },
                            {
                                "step": 1,
                                "phase": "residual_update",
                                "base_loss": "1.00000000",
                                "residual_loss": "0.75000000",
                                "hep_alpha": "",
                                "hep_loss": "",
                                "max_hep_logit_delta_from_ordinary": "",
                                "status": "ok",
                            },
                        ]
                    )
                return {
                    "experiment_id": experiment_id,
                    "status": "ok",
                    "error": None,
                    "phase0": {
                        "residual_objective": objective,
                        "training_steps": 1,
                        "base_loss": 1.0,
                        "zero_init_loss": 1.0,
                        "hep_alpha_sweep": (
                            [
                                {
                                    "alpha": 0.0,
                                    "loss": 0.75,
                                    "max_logit_delta_from_ordinary": 0.0,
                                }
                            ]
                            if experiment_id == "b"
                            else []
                        ),
                        "invariants": {"zero_init_identity": True},
                    },
                }

            with patch("relaleap.experiments.compare.run", side_effect=fake_run):
                comparison = run_comparison(configs, tmp_path / "comparison")

            self.assertEqual(comparison["status"], "ok")
            self.assertTrue((tmp_path / "comparison" / "summary.json").is_file())
            self.assertTrue((tmp_path / "comparison" / "metrics.csv").is_file())
            self.assertTrue((tmp_path / "comparison" / "notes.md").is_file())

            saved = json.loads(
                (tmp_path / "comparison" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(saved["runs"]), 2)
            self.assertEqual(saved["runs"][0]["residual_loss_delta"], -0.25)

            with (tmp_path / "comparison" / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("loss_delta_from_initial", rows[0])
            self.assertIn("hep_alpha", rows[0])
            self.assertIn("hep_loss", rows[0])
            self.assertIn("max_hep_logit_delta_from_ordinary", rows[0])
            self.assertEqual(rows[-1]["loss_delta_from_initial"], "-0.25000000")
            notes = (tmp_path / "comparison" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("## HEP Alpha Sweeps", notes)
            self.assertIn("alpha 0.0", notes)


if __name__ == "__main__":
    unittest.main()
