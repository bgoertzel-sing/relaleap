from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_same_student_intervention_matrix import (
    INCOMPLETE_MATRIX,
    run_causal_contextual_router_same_student_intervention_matrix,
)


class CausalContextualRouterSameStudentInterventionMatrixTest(unittest.TestCase):
    def test_available_matrix_passes_but_requires_token_position_null_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed)
                dirs.append(path)

            summary = run_causal_contextual_router_same_student_intervention_matrix(
                local_audit_dirs=dirs,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], INCOMPLETE_MATRIX)
            self.assertEqual(
                summary["selected_next_step"],
                "extend_distillation_agreement_audit_with_token_position_null_forced_support",
            )
            self.assertGreater(summary["key_metrics"]["teacher_forced_gain_all_tokens"], 0.0)
            self.assertFalse(
                summary["key_metrics"]["token_position_null_same_student_arm_available"]
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "same_student_matrix.csv").is_file())
            self.assertTrue((root / "report" / "seed_same_student_matrix.csv").is_file())
            self.assertTrue((root / "report" / "separate_student_null_reference.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_per_token_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed)
                (path / "per_token_supports.csv").unlink()
                dirs.append(path)

            summary = run_causal_contextual_router_same_student_intervention_matrix(
                local_audit_dirs=dirs,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(summary["failures"])


def _write_artifact(path: Path, *, seed: int) -> None:
    path.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "teacher_student_support_agreement_intervention_blocks_promotion",
        "claim_status": "distilled_causal_router_mechanism_not_established",
        "git_commit": f"seed-{seed}",
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
    _write_per_token_rows(path / "per_token_supports.csv")
    _write_null_rows(path / "null_control_metrics.csv")


def _write_per_token_rows(path: Path) -> None:
    fieldnames = [
        "fold",
        "flat_position",
        "target_token",
        "teacher_support",
        "student_support",
        "oracle_support",
        "student_router_support_loss",
        "teacher_support_forced_into_student_loss",
        "oracle_best_support_for_student_loss",
        "linear_support_forced_into_student_loss",
        "marginal_shuffled_student_support_loss",
        "uniform_random_support_loss",
        "teacher_student_exact_pair_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold in range(4):
            for position in range(5):
                writer.writerow(
                    {
                        "fold": fold,
                        "flat_position": position,
                        "target_token": position,
                        "teacher_support": "1,2",
                        "student_support": "1,2" if position % 2 else "3,4",
                        "oracle_support": "1,2",
                        "student_router_support_loss": 2.0,
                        "teacher_support_forced_into_student_loss": 1.8,
                        "oracle_best_support_for_student_loss": 1.7,
                        "linear_support_forced_into_student_loss": 2.5,
                        "marginal_shuffled_student_support_loss": 2.6,
                        "uniform_random_support_loss": 2.7,
                        "teacher_student_exact_pair_match": position % 2 != 0,
                    }
                )


def _write_null_rows(path: Path) -> None:
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
            writer.writerow(
                {
                    "fold": fold,
                    "null_control": "causal_distilled_from_token_position_frequency_matched_teacher_0.05",
                    "null_control_kind": "token_position_frequency_matched_teacher",
                    "student_minus_null_router_loss": 0.01,
                    "student_minus_null_oracle_regret": 0.01,
                    "student_minus_null_teacher_exact_pair_agreement": 0.1,
                }
            )


if __name__ == "__main__":
    unittest.main()
