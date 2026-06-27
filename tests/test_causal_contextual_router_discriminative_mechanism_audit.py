from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_discriminative_mechanism_audit import (
    DIRECT_CE_CAUSAL,
    FREQUENCY_NULL,
    INSUFFICIENT,
    REAL_CONTROL,
    SHUFFLED_NULL,
    SUPPORTED,
    run_causal_contextual_router_discriminative_mechanism_audit,
)


class CausalContextualRouterDiscriminativeMechanismAuditTest(unittest.TestCase):
    def test_passes_conservative_discriminative_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_dirs = []
            runpod_dirs = []
            for seed in [1, 2, 3]:
                local_dir = root / f"local_seed{seed}"
                runpod_dir = root / f"runpod_seed{seed}"
                _write_artifact(local_dir, seed=seed, backend="local")
                _write_artifact(runpod_dir, seed=seed, backend="runpod")
                local_dirs.append(local_dir)
                runpod_dirs.append(runpod_dir)
            synthesis = root / "synthesis.json"
            synthesis.write_text(
                json.dumps({"status": "pass", "claim_status": "supported"}) + "\n",
                encoding="utf-8",
            )
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Run discriminative mechanism control audit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_discriminative_mechanism_audit(
                local_audit_dirs=local_dirs,
                runpod_audit_dirs=runpod_dirs,
                synthesis_summary_path=synthesis,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], SUPPORTED)
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertTrue(
                summary["gate_status"]["passes_discriminative_mechanism_gate"]
            )
            self.assertEqual(len(summary["source_rows"]), 6)
            self.assertEqual(len(summary["paired_control_rows"]), 72)
            self.assertEqual(len(summary["unavailable_control_rows"]), 1)
            self.assertIn(
                "rank-matched or dense-matched residual control",
                summary["claim_boundaries"]["not_supported"],
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_artifacts.csv").is_file())
            self.assertTrue((root / "report" / "control_summary.csv").is_file())
            self.assertTrue((root / "report" / "fold_paired_controls.csv").is_file())
            self.assertTrue((root / "report" / "intervention_summary.csv").is_file())
            self.assertTrue((root / "report" / "unavailable_controls.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_synthesis_missing_or_not_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_dirs = []
            runpod_dirs = []
            for seed in [1, 2, 3]:
                local_dir = root / f"local_seed{seed}"
                runpod_dir = root / f"runpod_seed{seed}"
                _write_artifact(local_dir, seed=seed, backend="local")
                _write_artifact(runpod_dir, seed=seed, backend="runpod")
                local_dirs.append(local_dir)
                runpod_dirs.append(runpod_dir)

            summary = run_causal_contextual_router_discriminative_mechanism_audit(
                local_audit_dirs=local_dirs,
                runpod_audit_dirs=runpod_dirs,
                synthesis_summary_path=root / "missing-synthesis.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["claim_status"], INSUFFICIENT)
            self.assertTrue(
                any(
                    failure["field"] == "prior_synthesis_passed"
                    for failure in summary["failures"]
                )
            )


def _write_artifact(path: Path, *, seed: int, backend: str) -> None:
    path.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "teacher_student_support_agreement_intervention_supported",
        "claim_status": (
            "distilled_causal_router_support_mechanism_and_null_controls_supported_not_promoted"
        ),
        "git_commit": f"{backend}-{seed}",
        "audit": {
            "fold_count": 4,
            "dataset": "tiny_shakespeare_word",
            "support_router": "contextual_mlp_causal",
            "top_k": 2,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_fold_metrics(path / "fold_metrics.csv")
    _write_aggregate_metrics(path / "aggregate_metrics.csv")
    _write_intervention_metrics(path / "intervention_metrics.csv")
    for name in [
        "agreement_metrics.csv",
        "null_control_metrics.csv",
        "per_token_supports.csv",
        "support_counts.csv",
    ]:
        (path / name).write_text("placeholder\n", encoding="utf-8")
    (path / "notes.md").write_text("placeholder\n", encoding="utf-8")


def _write_fold_metrics(path: Path) -> None:
    fieldnames = [
        "fold",
        "control",
        "router_loss",
        "oracle_support_regret",
        "used_columns",
        "unique_support_sets",
        "support_load_entropy",
    ]
    values = {
        REAL_CONTROL: (2.8, 0.01, 23, 50, 0.94),
        SHUFFLED_NULL: (3.1, 0.31, 22, 48, 0.91),
        FREQUENCY_NULL: (3.12, 0.33, 22, 48, 0.90),
        DIRECT_CE_CAUSAL: (2.95, 0.14, 22, 45, 0.88),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold in range(4):
            for control, metrics in values.items():
                writer.writerow(
                    {
                        "fold": fold,
                        "control": control,
                        "router_loss": metrics[0] + fold * 0.001,
                        "oracle_support_regret": metrics[1] + fold * 0.001,
                        "used_columns": metrics[2],
                        "unique_support_sets": metrics[3],
                        "support_load_entropy": metrics[4],
                    }
                )


def _write_aggregate_metrics(path: Path) -> None:
    fieldnames = [
        "control",
        "folds",
        "mean_router_loss",
        "mean_oracle_support_regret",
        "mean_used_columns",
        "mean_unique_support_sets",
        "mean_support_load_entropy",
        "mean_support_change_fraction",
    ]
    values = {
        REAL_CONTROL: (2.8, 0.01, 23, 50, 0.94, 0.98),
        SHUFFLED_NULL: (3.1, 0.31, 22, 48, 0.91, 0.99),
        FREQUENCY_NULL: (3.12, 0.33, 22, 48, 0.90, 0.99),
        DIRECT_CE_CAUSAL: (2.95, 0.14, 22, 45, 0.88, 1.0),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for control, metrics in values.items():
            writer.writerow(
                {
                    "control": control,
                    "folds": 4,
                    "mean_router_loss": metrics[0],
                    "mean_oracle_support_regret": metrics[1],
                    "mean_used_columns": metrics[2],
                    "mean_unique_support_sets": metrics[3],
                    "mean_support_load_entropy": metrics[4],
                    "mean_support_change_fraction": metrics[5],
                }
            )


def _write_intervention_metrics(path: Path) -> None:
    fieldnames = [
        "fold",
        "token_subset",
        "intervention",
        "loss",
        "delta_vs_student_router_support",
    ]
    values = {
        "teacher_support_forced_into_student": -0.005,
        "oracle_best_support_for_student": -0.01,
        "linear_support_forced_into_student": 1.0,
        "marginal_shuffled_student_support": 0.7,
        "uniform_random_support": 0.8,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold in range(4):
            for intervention, delta in values.items():
                writer.writerow(
                    {
                        "fold": fold,
                        "token_subset": "all_tokens",
                        "intervention": intervention,
                        "loss": 2.8 + delta,
                        "delta_vs_student_router_support": delta,
                    }
                )


if __name__ == "__main__":
    unittest.main()
