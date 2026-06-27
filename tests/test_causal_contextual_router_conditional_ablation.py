from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_conditional_ablation import (
    TARGET_POSITION_CONFOUND,
    run_causal_contextual_router_conditional_ablation,
)


class CausalContextualRouterConditionalAblationTest(unittest.TestCase):
    def test_target_position_lookup_dominates_causal_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed)
                dirs.append(path)

            summary = run_causal_contextual_router_conditional_ablation(
                local_audit_dirs=dirs,
                out_dir=root / "report",
                min_target_position_edge=0.05,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], TARGET_POSITION_CONFOUND)
            self.assertEqual(
                summary["selected_next_step"],
                "conditional_permutation_resample_matrix_before_runpod_repeat",
            )
            self.assertGreater(
                summary["key_comparisons"][
                    "target_position_minus_causal_history_teacher_agreement"
                ],
                0.05,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "feature_ablation_metrics.csv").is_file())
            self.assertTrue((root / "report" / "seed_feature_metrics.csv").is_file())
            self.assertTrue((root / "report" / "token_position_gain_slices.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_per_token_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed in [1, 2, 3]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed)
                (path / "per_token_supports.csv").unlink()
                dirs.append(path)

            summary = run_causal_contextual_router_conditional_ablation(
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
    (path / "null_control_metrics.csv").write_text(
        "fold,null_control_kind,student_minus_null_router_loss\n",
        encoding="utf-8",
    )
    (path / "null_sampling_diagnostics.csv").write_text(
        "fold,sampling_mode,candidate_count\n",
        encoding="utf-8",
    )


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
        "teacher_student_exact_pair_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        patterns = {
            0: [0, 0, 1, 0, 1, 1, 0, 1],
            1: [1, 0, 0, 1, 0, 1, 1, 0],
            2: [0, 0, 1, 0, 1, 1, 0, 1],
            3: [1, 0, 0, 1, 0, 1, 1, 0],
        }
        for fold in range(4):
            for position in range(8):
                target = patterns[fold][position]
                support = f"{target},{target + 10}"
                causal_history_decoy = f"{(position + fold) % 4},{20 + ((position + fold) % 4)}"
                writer.writerow(
                    {
                        "fold": fold,
                        "flat_position": position,
                        "target_token": target,
                        "teacher_support": support,
                        "student_support": support if position % 3 else causal_history_decoy,
                        "oracle_support": support,
                        "student_router_support_loss": 2.0,
                        "teacher_support_forced_into_student_loss": 1.9,
                        "oracle_best_support_for_student_loss": 1.8,
                        "teacher_student_exact_pair_match": position % 3 != 0,
                    }
                )


if __name__ == "__main__":
    unittest.main()
