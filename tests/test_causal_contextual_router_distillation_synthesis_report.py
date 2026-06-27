from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_distillation_synthesis_report import (
    EXPECTED_FILES,
    INSUFFICIENT,
    NEXT_MECHANISM_AUDIT,
    SUPPORTED,
    run_causal_contextual_router_distillation_synthesis_report,
)


class CausalContextualRouterDistillationSynthesisReportTest(unittest.TestCase):
    def test_passes_cross_seed_gate_and_blocks_promotion(self) -> None:
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
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        (
                            "recommended_next_action: Add and run the cross-seed "
                            "causal-router distillation synthesis report"
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_distillation_synthesis_report(
                local_audit_dirs=local_dirs,
                runpod_audit_dirs=runpod_dirs,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], SUPPORTED)
            self.assertEqual(summary["selected_next_step"], NEXT_MECHANISM_AUDIT)
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(len(summary["source_rows"]), 6)
            self.assertEqual(len(summary["delta_rows"]), 12)
            self.assertEqual(len(summary["fold_delta_rows"]), 48)
            self.assertTrue(summary["gate_status"]["passes_synthesis_gate"])
            self.assertIn(
                "deployable causal router default",
                summary["claim_boundaries"]["not_supported"],
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_artifacts.csv").is_file())
            self.assertTrue((root / "report" / "seed_backend_deltas.csv").is_file())
            self.assertTrue((root / "report" / "fold_paired_deltas.csv").is_file())
            self.assertTrue((root / "report" / "backend_reproducibility.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_expected_artifact_missing(self) -> None:
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
            (local_dirs[0] / "per_token_supports.csv").unlink()

            summary = run_causal_contextual_router_distillation_synthesis_report(
                local_audit_dirs=local_dirs,
                runpod_audit_dirs=runpod_dirs,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["claim_status"], INSUFFICIENT)
            self.assertTrue(summary["failures"])


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
            "null_control_aggregates": {
                "causal_distilled_from_frequency_matched_teacher_0.05": _aggregate(
                    "frequency_matched_teacher"
                ),
                "causal_distilled_from_shuffled_teacher_0.05": _aggregate(
                    "shuffled_teacher"
                ),
            },
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in EXPECTED_FILES:
        target = path / name
        if target.exists():
            continue
        if name == "null_control_metrics.csv":
            _write_null_control_metrics(target)
        elif name.endswith(".csv"):
            target.write_text("placeholder\n", encoding="utf-8")
        elif name == "summary.json":
            continue
        else:
            target.write_text("placeholder\n", encoding="utf-8")


def _aggregate(kind: str) -> dict[str, object]:
    return {
        "folds": 4,
        "null_control": f"causal_distilled_from_{kind}_0.05",
        "mean_student_minus_null_router_loss": -0.2,
        "mean_student_minus_null_oracle_regret": -0.2,
        "mean_student_minus_null_teacher_exact_pair_agreement": 0.8,
    }


def _write_null_control_metrics(path: Path) -> None:
    fieldnames = [
        "fold",
        "null_control",
        "null_control_kind",
        "student_minus_null_router_loss",
        "student_minus_null_oracle_regret",
        "student_minus_null_teacher_exact_pair_agreement",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold in range(4):
            for kind in ["frequency_matched_teacher", "shuffled_teacher"]:
                writer.writerow(
                    {
                        "fold": fold,
                        "null_control": f"causal_distilled_from_{kind}_0.05",
                        "null_control_kind": kind,
                        "student_minus_null_router_loss": "-0.2",
                        "student_minus_null_oracle_regret": "-0.2",
                        "student_minus_null_teacher_exact_pair_agreement": "0.8",
                    }
                )


if __name__ == "__main__":
    unittest.main()
