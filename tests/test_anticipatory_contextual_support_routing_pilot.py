from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_pilot import (
    REQUIRED_ARTIFACTS,
    run_anticipatory_contextual_support_routing_pilot_contract,
)


class AnticipatoryContextualSupportRoutingPilotContractTest(unittest.TestCase):
    def test_contract_records_existing_pilot_and_major_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "audit"
            dense = root / "dense.json"
            review = root / "latest-review.md"
            _write_audit_packet(audit_dir)
            _write_dense_synthesis(dense)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Pivot to ACSR pilot locally, no GPU",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_anticipatory_contextual_support_routing_pilot_contract(
                audit_dir=audit_dir,
                dense_synthesis_path=dense,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "anticipatory_contextual_support_routing_pilot_contract_recorded",
            )
            self.assertEqual(
                summary["claim_status"],
                "acsr_pilot_artifact_contract_satisfied_not_promoted",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            self.assertIn("Ben should be notified", summary["direction_shift"]["record"])
            self.assertIn("source-of-truth ACSR pilot artifacts already exist", summary["strategy_review_handling"])
            self.assertTrue(all(row["passed"] for row in summary["gate_criteria"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_contract_fails_closed_when_required_router_arm_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = root / "audit"
            dense = root / "dense.json"
            review = root / "latest-review.md"
            _write_audit_packet(audit_dir, omit_variant="random_fixed_topk2")
            _write_dense_synthesis(dense)
            review.write_text(
                "strategic_change_level: minor\nnotify_ben: false\n",
                encoding="utf-8",
            )

            summary = run_anticipatory_contextual_support_routing_pilot_contract(
                audit_dir=audit_dir,
                dense_synthesis_path=dense,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "anticipatory_contextual_support_routing_pilot_contract_failed_closed",
            )
            self.assertTrue(
                any(
                    row["criterion"] == "required_router_arms_present"
                    and not row["passed"]
                    for row in summary["failures"]
                )
            )
            self.assertTrue((root / "out" / "summary.json").is_file())


def _write_audit_packet(path: Path, *, omit_variant: str | None = None) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "anticipatory_contextual_support_routing_smoke_completed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    variants = [
        "full_context_contextual_topk2_teacher",
        "causal_feature_safe_contextual_topk2",
        "acsr_mlp_predicted_future",
        "shuffled_predicted_features",
        "token_position_only_predicted_features",
        "random_fixed_topk2",
        "rank_matched_contextual_topk1",
        "parameter_matched_causal_mlp_control",
    ]
    _write_csv(
        path / "router_metrics.csv",
        [
            {
                "variant": variant,
                "ce_loss": 2.0 if variant == "acsr_mlp_predicted_future" else 2.2,
                "oracle_regret": 0.1,
            }
            for variant in variants
            if variant != omit_variant
        ],
    )
    _write_csv(
        path / "same_student_metrics.csv",
        [
            {
                "forcing_type": "same_student",
                "status": "available",
                "comparison": "acsr_mlp_predicted_future_support_vs_shuffled_predicted_features",
            },
            {
                "forcing_type": "same_student",
                "status": "available",
                "comparison": "acsr_mlp_predicted_future_support_vs_token_position_only_predicted_features",
            },
        ],
    )
    _write_csv(
        path / "feature_perturbation.csv",
        [
            {
                "control_type": "future_perturbation_negative",
                "passed": True,
            },
            {
                "control_type": "leaky_future_positive",
                "passed": True,
            },
        ],
    )
    _write_csv(
        path / "retention_churn_metrics.csv",
        [{"phase": "second_context_transfer", "variant": "acsr_mlp_predicted_future"}],
    )
    _write_csv(
        path / "parameter_counts.csv",
        [
            {
                "component": "parameter_matched_causal_mlp_control",
                "status": "available",
            }
        ],
    )


def _write_dense_synthesis(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "acsr_sparse_support_claim_blocked_by_dense_rank_norm_controls",
                "claim_status": "dense_rank16_24_controls_explain_ce_gain_threshold",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
