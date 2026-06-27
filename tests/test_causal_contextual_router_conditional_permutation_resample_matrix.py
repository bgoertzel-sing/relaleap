from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_conditional_permutation_resample_matrix import (
    ASSIGNMENT_ONLY,
    run_causal_contextual_router_conditional_permutation_resample_matrix,
)


class CausalContextualRouterConditionalPermutationResampleMatrixTest(unittest.TestCase):
    def test_assignment_signal_writes_report_but_functional_gate_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed, include_token_position_null=True)
                dirs.append(path)

            summary = run_causal_contextual_router_conditional_permutation_resample_matrix(
                local_audit_dirs=dirs,
                out_dir=root / "report",
                resamples=64,
                random_seed=11,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], ASSIGNMENT_ONLY)
            self.assertEqual(
                summary["selected_next_step"],
                "keep_causal_router_distillation_promotion_frozen",
            )
            self.assertTrue(summary["gate_status"]["assignment_gate_passes"])
            self.assertFalse(summary["gate_status"]["functional_gate_passes"])
            self.assertLessEqual(
                summary["key_metrics"]["student_exact_agreement_empirical_p_upper"],
                0.05,
            )
            self.assertLess(
                summary["key_metrics"]["teacher_minus_token_position_null_gain_all_tokens"],
                0.0,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "conditional_resample_metrics.csv").is_file())
            self.assertTrue((root / "report" / "functional_same_student_reference.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_token_position_same_student_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed, include_token_position_null=False)
                dirs.append(path)

            summary = run_causal_contextual_router_conditional_permutation_resample_matrix(
                local_audit_dirs=dirs,
                out_dir=root / "report",
                resamples=8,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(summary["failures"])


def _write_artifact(
    path: Path,
    *,
    seed: int,
    include_token_position_null: bool,
) -> None:
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
    _write_per_token_rows(
        path / "per_token_supports.csv",
        include_token_position_null=include_token_position_null,
    )
    (path / "null_control_metrics.csv").write_text(
        "fold,null_control_kind,student_minus_null_router_loss\n",
        encoding="utf-8",
    )
    (path / "null_sampling_diagnostics.csv").write_text(
        "fold,sampling_mode,candidate_count\n",
        encoding="utf-8",
    )


def _write_per_token_rows(
    path: Path,
    *,
    include_token_position_null: bool,
) -> None:
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
        "teacher_student_exact_pair_match",
    ]
    if include_token_position_null:
        fieldnames.extend(
            [
                "token_position_null_support",
                "token_position_null_support_forced_into_student_loss",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold in range(4):
            for position in range(8):
                target = position % 2
                support = f"{position + 10 * fold},{position + 10 * fold + 100}"
                null_support = f"{(position + fold + 1) % 8},{20 + ((position + fold + 1) % 8)}"
                row = {
                    "fold": fold,
                    "flat_position": position,
                    "target_token": target,
                    "teacher_support": support,
                    "student_support": support,
                    "oracle_support": support,
                    "student_router_support_loss": 2.0,
                    "teacher_support_forced_into_student_loss": 2.1,
                    "oracle_best_support_for_student_loss": 1.8,
                    "teacher_student_exact_pair_match": "true",
                }
                if include_token_position_null:
                    row.update(
                        {
                            "token_position_null_support": null_support,
                            "token_position_null_support_forced_into_student_loss": 2.0,
                        }
                    )
                writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
