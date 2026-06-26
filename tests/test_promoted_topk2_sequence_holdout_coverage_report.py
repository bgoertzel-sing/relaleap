from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_sequence_holdout_coverage_report import (
    SEQUENCE_HOLDOUT_COVERAGE_READY,
    SEQUENCE_HOLDOUT_EXTENSION_REQUIRED,
    SEQUENCE_HOLDOUT_SUPPORT_HEAD_GENERALIZATION_FAILED,
    run_promoted_topk2_sequence_holdout_coverage_report,
)


class PromotedTopk2SequenceHoldoutCoverageReportTest(unittest.TestCase):
    def test_records_missing_sequence_holdout_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = _write_sources(root, sequence_split=False)
            review = _write_review(root)

            summary = run_promoted_topk2_sequence_holdout_coverage_report(
                support_selection_dir=dirs["support"],
                exhaustive_audit_dir=dirs["audit"],
                causal_adequacy_dir=dirs["causal"],
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SEQUENCE_HOLDOUT_EXTENSION_REQUIRED)
            self.assertTrue(summary["signals"]["position_holdout_present"])
            self.assertFalse(summary["signals"]["sequence_level_holdout_present"])
            self.assertTrue(
                summary["signals"][
                    "deployable_support_selection_claim_blocked_by_split_coverage"
                ]
            )
            self.assertIn("support_audit", summary["next_step"])
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue((root / "out" / "split_rows.csv").is_file())
            self.assertTrue((root / "out" / "source_rows.csv").is_file())
            self.assertTrue((root / "out" / "notes.md").is_file())

    def test_accepts_existing_sequence_holdout_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = _write_sources(root, sequence_split=True)

            summary = run_promoted_topk2_sequence_holdout_coverage_report(
                support_selection_dir=dirs["support"],
                exhaustive_audit_dir=dirs["audit"],
                causal_adequacy_dir=dirs["causal"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SEQUENCE_HOLDOUT_COVERAGE_READY)
            self.assertTrue(summary["signals"]["sequence_level_holdout_present"])
            self.assertFalse(
                summary["signals"][
                    "deployable_support_selection_claim_blocked_by_split_coverage"
                ]
            )

    def test_blocks_failed_sequence_support_head_generalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dirs = _write_sources(
                root,
                sequence_split=True,
                sequence_support_head_failed=True,
            )

            summary = run_promoted_topk2_sequence_holdout_coverage_report(
                support_selection_dir=dirs["support"],
                exhaustive_audit_dir=dirs["audit"],
                causal_adequacy_dir=dirs["causal"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                SEQUENCE_HOLDOUT_SUPPORT_HEAD_GENERALIZATION_FAILED,
            )
            self.assertTrue(
                summary["signals"]["sequence_support_head_generalization_failed"]
            )
            self.assertTrue(
                summary["signals"]["sequence_recovery_fraction_numerically_fragile"]
            )
            self.assertEqual(
                summary["metrics"][
                    "sequence_support_head_intervention_minus_router_loss"
                ],
                0.02,
            )
            self.assertIn("K-fold", summary["next_step"])


def _write_sources(
    root: Path,
    *,
    sequence_split: bool,
    sequence_support_head_failed: bool = False,
) -> dict[str, Path]:
    dirs = {
        "support": root / "support",
        "audit": root / "audit",
        "causal": root / "causal",
    }
    for path in dirs.values():
        path.mkdir(parents=True)
    _write_json(
        dirs["support"] / "summary.json",
        {
            "status": "pass",
            "decision": "promoted_topk2_support_selection_quality_established",
            "metrics": {
                "config_path": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
                "dataset": "tiny_shakespeare_word",
                "support_router": "contextual_mlp",
                "contextual_support_head_holdout_gap_recovery": 0.19,
                "contextual_oracle_target_holdout_gap_recovery": 1.0,
                "oracle_support_regret": 0.0025,
            },
        },
    )
    holdout_split = (
        "held-out full sequences"
        if sequence_split
        else "odd flattened token positions"
    )
    train_split = (
        "training full sequences"
        if sequence_split
        else "even flattened token positions"
    )
    audit = {
        "status": "ok",
        "audit": {
            "config_path": "configs/token_larger_support_wide_contextual_router_hep_temporal_clipped_objective_gate.yaml",
            "dataset": "tiny_shakespeare_word",
            "support_router": "contextual_mlp",
            "contextual_router_support_head": {
                "train_split": train_split,
                "holdout_split": holdout_split,
                "holdout": {"oracle_gap_recovery_fraction": 0.19},
            },
            "router_oracle_target_contextual_diagnostic": {
                "train_split": train_split,
                "holdout_split": holdout_split,
                "holdout": {
                    "oracle_gap_recovery_fraction": 1.0,
                    "router_loss": 2.87,
                    "oracle_loss": 2.865,
                    "selector_loss": 2.865,
                    "selector_minus_router_loss": -0.005,
                    "selector_oracle_regret": 0.0,
                },
            },
        },
    }
    if sequence_support_head_failed:
        audit["audit"]["contextual_router_support_sequence_head"] = {
            "train_split": "even full sequences",
            "holdout_split": "odd full sequences",
            "holdout": {
                "oracle_gap_recovery_fraction": -4.0,
                "router_loss": 2.87,
                "oracle_loss": 2.865,
                "intervention_loss": 2.89,
                "intervention_minus_router_loss": 0.02,
                "intervention_oracle_regret": 0.025,
            },
        }
    _write_json(dirs["audit"] / "summary.json", audit)
    _write_json(
        dirs["causal"] / "summary.json",
        {
            "status": "pass",
            "decision": "predictive_default_causal_adequacy_not_established",
            "metrics": {
                "topk2_support_churn": 0.86,
                "topk2_to_topk1_finite_update_logit_mse_ratio": 25.8,
            },
        },
    )
    return dirs


def _write_review(root: Path) -> Path:
    path = root / "latest-review.md"
    path.write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: Add a stricter sequence-level held-out evaluation.",
                "",
                "Include sequence-level holdout before deployable support claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
