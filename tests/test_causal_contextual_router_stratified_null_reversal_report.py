from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.causal_contextual_router_stratified_null_reversal_report import (
    CLAIM_NOT_ESTABLISHED,
    EXPECTED_FILES,
    run_causal_contextual_router_stratified_null_reversal_report,
)


class CausalContextualRouterStratifiedNullReversalReportTest(unittest.TestCase):
    def test_writes_reversal_report_and_blocks_prior_mechanism_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed, delta in [(1, -0.012), (2, -0.003), (3, 0.001)]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed, ce_delta=delta)
                dirs.append(path)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        (
                            "recommended_next_action: Freeze causal-router "
                            "distillation promotion"
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_causal_contextual_router_stratified_null_reversal_report(
                local_audit_dirs=dirs,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["claim_status"], CLAIM_NOT_ESTABLISHED)
            self.assertTrue(summary["strategy_review"]["notify_ben"])
            self.assertEqual(
                summary["selected_next_step"],
                "conditional_token_position_vs_context_ablation_before_runpod_repeat",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "fold_stratified_null_deltas.csv").is_file())
            self.assertTrue((root / "report" / "null_sampling_summary.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_sampling_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = []
            for seed, delta in [(1, -0.012), (2, -0.003), (3, 0.001)]:
                path = root / f"seed{seed}"
                _write_artifact(path, seed=seed, ce_delta=delta)
                (path / "null_sampling_diagnostics.csv").unlink()
                dirs.append(path)

            summary = run_causal_contextual_router_stratified_null_reversal_report(
                local_audit_dirs=dirs,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertTrue(summary["failures"])


def _write_artifact(path: Path, *, seed: int, ce_delta: float) -> None:
    path.mkdir(parents=True)
    null_name = "causal_distilled_from_token_position_frequency_matched_teacher_0.05"
    summary = {
        "status": "pass",
        "decision": "teacher_student_support_agreement_intervention_blocks_promotion",
        "claim_status": "distilled_causal_router_mechanism_not_established",
        "audit": {
            "fold_count": 4,
            "dataset": "tiny_shakespeare_word",
            "support_router": "contextual_mlp_causal",
            "top_k": 2,
            "null_control_aggregates": {
                null_name: {
                    "null_control": null_name,
                    "mean_student_minus_null_router_loss": ce_delta,
                    "mean_student_minus_null_oracle_regret": ce_delta,
                    "mean_student_minus_null_teacher_exact_pair_agreement": 0.14,
                }
            },
            "null_sampling_aggregates": {
                null_name: {
                    "positions": 192,
                    "target_position_fraction": 0.5,
                    "target_only_fraction": 0.4,
                    "global_fraction": 0.1,
                    "mean_candidate_count": 3.0,
                    "mean_original_stratum_size": 4.0,
                    "mean_candidate_support_entropy": 0.7,
                }
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
            _write_null_control_metrics(target, null_name, ce_delta)
        elif name == "null_sampling_diagnostics.csv":
            target.write_text(
                "fold,null_control,sampling_mode,candidate_count\n"
                f"0,{null_name},target_position,3\n",
                encoding="utf-8",
            )
        elif name.endswith(".csv"):
            target.write_text("placeholder\n", encoding="utf-8")
        elif name != "summary.json":
            target.write_text("placeholder\n", encoding="utf-8")


def _write_null_control_metrics(path: Path, null_name: str, ce_delta: float) -> None:
    fieldnames = [
        "fold",
        "null_control",
        "null_control_kind",
        "student_minus_null_router_loss",
        "student_minus_null_oracle_regret",
        "student_minus_null_teacher_exact_pair_agreement",
    ]
    signs = [-1.0, 1.0, -1.0, 1.0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for fold, sign in enumerate(signs):
            writer.writerow(
                {
                    "fold": fold,
                    "null_control": null_name,
                    "null_control_kind": "token_position_frequency_matched_teacher",
                    "student_minus_null_router_loss": ce_delta + 0.004 * sign,
                    "student_minus_null_oracle_regret": ce_delta + 0.004 * sign,
                    "student_minus_null_teacher_exact_pair_agreement": 0.14,
                }
            )


if __name__ == "__main__":
    unittest.main()
