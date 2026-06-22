from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.compare import (
    DEFAULT_CONFIGS,
    _comparison_baseline,
    _comparison_entry,
    _comparison_verdict,
    compare_comparison_to_baseline,
    compare_to_baseline,
    run_comparison,
    write_comparison_baseline,
)


def _passing_artifact_invariants() -> dict[str, bool]:
    return {
        "summary_json": True,
        "metrics_csv": True,
        "notes_md": True,
    }


class ComparisonReportTest(unittest.TestCase):
    def test_comparison_entry_summarizes_loss_trajectory(self) -> None:
        summary = {
            "experiment_id": "char_smoke",
            "status": "ok",
            "error": None,
            "phase0": {
                "residual_objective": "supervised_ce",
                "dataset": "tiny_shakespeare_word",
                "training_steps": 2,
                "base_loss": 3.0,
                "zero_init_loss": 3.0,
                "invariants": {"zero_init_identity": True},
            },
            "artifact_invariants": _passing_artifact_invariants(),
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
        self.assertEqual(entry["dataset"], "tiny_shakespeare_word")
        self.assertFalse(entry["pinned_support"])
        self.assertFalse(entry["support_stress"])
        self.assertEqual(entry["support_instability"], {})
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

    def test_comparison_verdict_summarizes_invariants_and_best_hep(self) -> None:
        verdict = _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke",
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "hep_alpha_sweep": [],
                },
                {
                    "experiment_id": "char_smoke_hep",
                    "invariants": {
                        "zero_init_identity": True,
                        "hep_alpha_0_equivalence": True,
                    },
                    "artifact_invariants": _passing_artifact_invariants(),
                    "hep_alpha_sweep": [
                        {
                            "alpha": 0.0,
                            "loss": 3.5,
                            "max_logit_delta_from_ordinary": 0.0,
                        },
                        {
                            "alpha": 0.5,
                            "loss": 3.25,
                            "max_logit_delta_from_ordinary": 0.01,
                        },
                    ],
                },
            ],
            "ok",
        )

        self.assertEqual(verdict["status"], "pass")
        self.assertTrue(verdict["invariants_passed"])
        self.assertEqual(verdict["invariant_count"], 3)
        self.assertEqual(verdict["failed_invariants"], [])
        self.assertEqual(
            verdict["best_hep_alpha_by_loss"],
            {
                "experiment_id": "char_smoke_hep",
                "alpha": 0.5,
                "loss": 3.25,
                "max_logit_delta_from_ordinary": 0.01,
            },
        )
        self.assertEqual(verdict["hep_alpha_acceptance"]["status"], "accepted")
        self.assertEqual(
            verdict["hep_alpha_acceptance"]["accepted_alpha"]["alpha"],
            0.5,
        )
        self.assertEqual(
            verdict["hep_alpha_acceptance"]["accepted_alpha"][
                "loss_improvement_from_alpha0"
            ],
            0.25,
        )

    def test_hep_acceptance_rejects_low_loss_alpha_over_delta_budget(self) -> None:
        verdict = _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke_hep",
                    "invariants": {
                        "zero_init_identity": True,
                        "hep_alpha_0_equivalence": True,
                    },
                    "artifact_invariants": _passing_artifact_invariants(),
                    "hep_alpha_sweep": [
                        {
                            "alpha": 0.0,
                            "loss": 3.50,
                            "max_logit_delta_from_ordinary": 0.0,
                        },
                        {
                            "alpha": 0.25,
                            "loss": 3.30,
                            "max_logit_delta_from_ordinary": 0.05,
                        },
                        {
                            "alpha": 0.5,
                            "loss": 3.20,
                            "max_logit_delta_from_ordinary": 0.20,
                        },
                    ],
                },
            ],
            "ok",
            hep_max_logit_delta=0.10,
            hep_min_loss_improvement=0.0,
        )

        self.assertEqual(verdict["best_hep_alpha_by_loss"]["alpha"], 0.5)
        self.assertEqual(verdict["hep_alpha_acceptance"]["status"], "accepted")
        self.assertEqual(
            verdict["hep_alpha_acceptance"]["accepted_alpha"]["alpha"],
            0.25,
        )
        rejected = [
            candidate
            for candidate in verdict["hep_alpha_acceptance"]["candidates"]
            if not candidate["accepted"]
        ]
        self.assertEqual([candidate["alpha"] for candidate in rejected], [0.5])

    def test_comparison_verdict_reports_failed_invariants(self) -> None:
        verdict = _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke",
                    "invariants": {
                        "zero_init_identity": True,
                        "frozen_base_unchanged": False,
                    },
                    "artifact_invariants": _passing_artifact_invariants(),
                    "hep_alpha_sweep": [],
                }
            ],
            "ok",
        )

        self.assertEqual(verdict["status"], "fail")
        self.assertFalse(verdict["invariants_passed"])
        self.assertEqual(
            verdict["failed_invariants"],
            [
                {
                    "experiment_id": "char_smoke",
                    "invariant": "frozen_base_unchanged",
                }
            ],
        )

    def test_comparison_verdict_fails_when_artifact_invariants_missing(self) -> None:
        verdict = _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke",
                    "invariants": {"zero_init_identity": True},
                    "hep_alpha_sweep": [],
                }
            ],
            "ok",
        )

        self.assertEqual(verdict["status"], "fail")
        self.assertTrue(verdict["invariants_passed"])
        self.assertFalse(verdict["artifact_invariants_passed"])
        self.assertEqual(verdict["artifact_invariant_count"], 3)
        self.assertEqual(
            verdict["failed_artifact_invariants"],
            [
                {"experiment_id": "char_smoke", "artifact": "summary_json"},
                {"experiment_id": "char_smoke", "artifact": "metrics_csv"},
                {"experiment_id": "char_smoke", "artifact": "notes_md"},
            ],
        )

    def test_comparison_verdict_reports_failed_artifact_invariants(self) -> None:
        verdict = _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke",
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": {
                        "summary_json": True,
                        "metrics_csv": False,
                        "notes_md": True,
                    },
                    "hep_alpha_sweep": [],
                }
            ],
            "ok",
        )

        self.assertEqual(verdict["status"], "fail")
        self.assertTrue(verdict["invariants_passed"])
        self.assertFalse(verdict["artifact_invariants_passed"])
        self.assertEqual(verdict["artifact_invariant_count"], 3)
        self.assertEqual(
            verdict["failed_artifact_invariants"],
            [
                {
                    "experiment_id": "char_smoke",
                    "artifact": "metrics_csv",
                }
            ],
        )

    def test_comparison_baseline_keeps_stable_phase0_fields(self) -> None:
        comparison = {
            "status": "ok",
            "verdict": _comparison_verdict(
                [
                    {
                        "experiment_id": "char_smoke_hep",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.5,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                            {
                                "alpha": 0.25,
                                "loss": 3.4,
                                "max_logit_delta_from_ordinary": 0.05,
                            },
                        ],
                    }
                ],
                "ok",
            ),
            "runs": [
                {
                    "experiment_id": "char_smoke_hep",
                    "config_path": "configs/char_smoke_hep.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.4,
                }
            ],
        }

        baseline = _comparison_baseline(comparison)

        self.assertEqual(baseline["schema_version"], 3)
        self.assertEqual(baseline["comparison_status"], "ok")
        self.assertEqual(baseline["verdict_status"], "pass")
        self.assertEqual(baseline["phase0_invariants"]["count"], 1)
        self.assertTrue(baseline["artifact_invariants"]["passed"])
        self.assertEqual(baseline["artifact_invariants"]["count"], 3)
        self.assertEqual(baseline["artifact_invariants"]["failed"], [])
        self.assertEqual(
            baseline["runs"][0]["artifact_invariants"],
            {"count": 3, "failed": [], "passed": True},
        )
        self.assertEqual(baseline["hep"]["best_alpha_by_loss"]["alpha"], 0.25)
        self.assertEqual(
            baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"],
            0.25,
        )
        self.assertEqual(
            baseline["hep"]["acceptance"]["accepted_alpha"][
                "loss_improvement_from_alpha0"
            ],
            0.10000000000000009,
        )
        self.assertNotIn("runtime_seconds", baseline)
        self.assertNotIn("out_dir", baseline)

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
                    "artifact_invariants": _passing_artifact_invariants(),
                    "phase0": {
                        "residual_objective": objective,
                        "dataset": "tiny_shakespeare_word",
                        "training_steps": 1,
                        "base_loss": 1.0,
                        "zero_init_loss": 1.0,
                        "pinned_support": experiment_id == "b",
                        "support_stress": experiment_id == "b",
                        "support_instability": {
                            "support_change_fraction": 0.25 if experiment_id == "b" else 0.0,
                            "pinned_vs_repicked_logit_delta": 1.5 if experiment_id == "b" else 0.0,
                        },
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
            self.assertEqual(saved["runs"][1]["dataset"], "tiny_shakespeare_word")
            self.assertTrue(saved["runs"][1]["pinned_support"])
            self.assertTrue(saved["runs"][1]["support_stress"])
            self.assertEqual(
                saved["runs"][1]["support_instability"]["support_change_fraction"],
                0.25,
            )
            self.assertEqual(saved["verdict"]["status"], "pass")
            self.assertTrue(saved["verdict"]["artifact_invariants_passed"])
            self.assertEqual(saved["verdict"]["artifact_invariant_count"], 6)
            self.assertEqual(saved["verdict"]["best_hep_alpha_by_loss"]["alpha"], 0.0)
            self.assertEqual(
                saved["verdict"]["hep_alpha_acceptance"]["status"],
                "no_nonzero_hep_candidates",
            )

            with (tmp_path / "comparison" / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("loss_delta_from_initial", rows[0])
            self.assertIn("hep_alpha", rows[0])
            self.assertIn("hep_loss", rows[0])
            self.assertIn("max_hep_logit_delta_from_ordinary", rows[0])
            self.assertIn("dataset", rows[0])
            self.assertEqual(rows[-1]["dataset"], "tiny_shakespeare_word")
            self.assertIn("pinned_support", rows[0])
            self.assertIn("support_stress", rows[0])
            self.assertEqual(rows[-1]["loss_delta_from_initial"], "-0.25000000")
            notes = (tmp_path / "comparison" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("## HEP Alpha Sweeps", notes)
            self.assertIn("alpha 0.0", notes)
            self.assertIn("Pinned-vs-repicked", notes)
            self.assertIn("Best HEP alpha by loss", notes)
            self.assertIn("Accepted HEP alpha", notes)

            baseline = write_comparison_baseline(
                tmp_path / "baseline" / "phase0.json",
                comparison,
            )
            self.assertTrue((tmp_path / "baseline" / "phase0.json").is_file())
            self.assertEqual(baseline["comparison_status"], "ok")
            self.assertEqual(baseline["phase0_invariants"]["count"], 2)
            self.assertEqual(baseline["artifact_invariants"]["count"], 6)
            self.assertEqual(
                baseline["hep"]["acceptance"]["status"],
                "no_nonzero_hep_candidates",
            )

    def test_checked_in_phase0_baseline_records_accepted_hep_alpha(self) -> None:
        baseline_path = Path("baselines/phase0_char_smoke_comparison.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        self.assertEqual(baseline["schema_version"], 3)
        self.assertEqual(baseline["comparison_status"], "ok")
        self.assertEqual(baseline["verdict_status"], "pass")
        self.assertEqual(
            baseline["config_paths"],
            [
                "configs/char_smoke.yaml",
                "configs/char_smoke_pc.yaml",
                "configs/char_smoke_hep.yaml",
            ],
        )
        self.assertTrue(baseline["phase0_invariants"]["passed"])
        self.assertEqual(baseline["phase0_invariants"]["count"], 12)
        self.assertTrue(baseline["artifact_invariants"]["passed"])
        self.assertEqual(baseline["artifact_invariants"]["count"], 9)
        self.assertEqual(baseline["hep"]["best_alpha_by_loss"]["alpha"], 1.0)
        self.assertEqual(
            baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"],
            0.25,
        )
        self.assertEqual(baseline["hep"]["acceptance"]["rejected_count"], 2)
        self.assertEqual(
            [
                entry["artifact_invariants"]
                for entry in baseline["runs"]
            ],
            [
                {"count": 3, "failed": [], "passed": True},
                {"count": 3, "failed": [], "passed": True},
                {"count": 3, "failed": [], "passed": True},
            ],
        )

    def test_compare_to_baseline_reports_acceptance_match(self) -> None:
        comparison = {
            "status": "ok",
            "verdict": _comparison_verdict(
                [
                    {
                        "experiment_id": "char_smoke_hep",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.5,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                            {
                                "alpha": 0.25,
                                "loss": 3.4,
                                "max_logit_delta_from_ordinary": 0.05,
                            },
                        ],
                    }
                ],
                "ok",
            ),
            "runs": [
                {
                    "experiment_id": "char_smoke_hep",
                    "config_path": "configs/char_smoke_hep.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.4,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            baseline_path = tmp_path / "baseline.json"
            out_path = tmp_path / "baseline_comparison.json"
            write_comparison_baseline(baseline_path, comparison)

            result = compare_to_baseline(comparison, baseline_path, out_path)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["mismatches"], [])
            self.assertEqual(
                result["candidate"]["accepted_hep_alpha"]["alpha"],
                0.25,
            )
            self.assertTrue(out_path.is_file())

    def test_compare_to_baseline_reports_schema_version_mismatch(self) -> None:
        comparison = {
            "status": "ok",
            "verdict": _comparison_verdict(
                [
                    {
                        "experiment_id": "char_smoke_hep",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.5,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                        ],
                    }
                ],
                "ok",
            ),
            "runs": [
                {
                    "experiment_id": "char_smoke_hep",
                    "config_path": "configs/char_smoke_hep.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.5,
                }
            ],
        }
        reference = _comparison_baseline(comparison)
        reference["schema_version"] = 1
        reference.pop("artifact_invariants")

        result = compare_comparison_to_baseline(comparison, reference)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(
            result["mismatches"],
            [
                {
                    "field": "schema_version",
                    "reference": 1,
                    "candidate": 3,
                }
            ],
        )
        self.assertIsNone(result["reference"]["artifact_invariants"])

    def test_compare_to_baseline_flags_accepted_alpha_drift(self) -> None:
        comparison = {
            "status": "ok",
            "verdict": _comparison_verdict(
                [
                    {
                        "experiment_id": "char_smoke_hep",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.5,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                            {
                                "alpha": 0.25,
                                "loss": 3.4,
                                "max_logit_delta_from_ordinary": 0.05,
                            },
                        ],
                    }
                ],
                "ok",
            ),
            "runs": [
                {
                    "experiment_id": "char_smoke_hep",
                    "config_path": "configs/char_smoke_hep.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.4,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            baseline_path = tmp_path / "baseline.json"
            out_path = tmp_path / "baseline_comparison.json"
            reference = write_comparison_baseline(baseline_path, comparison)
            reference["hep"]["acceptance"]["accepted_alpha"]["alpha"] = 0.5
            baseline_path.write_text(
                json.dumps(reference, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            result = compare_to_baseline(comparison, baseline_path, out_path)

            self.assertEqual(result["status"], "fail")
            self.assertEqual(
                result["mismatches"],
                [
                    {
                        "field": "hep.acceptance.accepted_alpha.alpha",
                        "reference": 0.5,
                        "candidate": 0.25,
                    }
                ],
            )

    def test_compare_to_baseline_flags_per_run_artifact_contract_drift(self) -> None:
        comparison = {
            "status": "ok",
            "verdict": _comparison_verdict(
                [
                    {
                        "experiment_id": "char_smoke",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [],
                    },
                    {
                        "experiment_id": "char_smoke_hep",
                        "invariants": {"zero_init_identity": True},
                        "artifact_invariants": _passing_artifact_invariants(),
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.5,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                        ],
                    },
                ],
                "ok",
            ),
            "runs": [
                {
                    "experiment_id": "char_smoke",
                    "config_path": "configs/char_smoke.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.5,
                },
                {
                    "experiment_id": "char_smoke_hep",
                    "config_path": "configs/char_smoke_hep.yaml",
                    "residual_objective": "supervised_ce",
                    "status": "ok",
                    "training_steps": 10,
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "final_residual_loss": 3.5,
                },
            ],
        }
        reference = _comparison_baseline(comparison)
        candidate = json.loads(json.dumps(comparison))
        del candidate["runs"][1]["artifact_invariants"]

        result = compare_comparison_to_baseline(candidate, reference)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(
            result["mismatches"],
            [
                {
                    "field": "runs.artifact_invariants",
                    "reference": [
                        {
                            "experiment_id": "char_smoke",
                            "config_path": "configs/char_smoke.yaml",
                            "artifact_invariants": {
                                "count": 3,
                                "failed": [],
                                "passed": True,
                            },
                        },
                        {
                            "experiment_id": "char_smoke_hep",
                            "config_path": "configs/char_smoke_hep.yaml",
                            "artifact_invariants": {
                                "count": 3,
                                "failed": [],
                                "passed": True,
                            },
                        },
                    ],
                    "candidate": [
                        {
                            "experiment_id": "char_smoke",
                            "config_path": "configs/char_smoke.yaml",
                            "artifact_invariants": {
                                "count": 3,
                                "failed": [],
                                "passed": True,
                            },
                        },
                        {
                            "experiment_id": "char_smoke_hep",
                            "config_path": "configs/char_smoke_hep.yaml",
                            "artifact_invariants": {
                                "count": 3,
                                "failed": [
                                    "summary_json",
                                    "metrics_csv",
                                    "notes_md",
                                ],
                                "passed": False,
                            },
                        },
                    ],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
