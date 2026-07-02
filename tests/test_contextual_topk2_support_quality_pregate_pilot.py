from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.contextual_topk2_support_quality_pregate_pilot import (
    INSUFFICIENT_EVIDENCE,
    PILOT_RECORDED,
    run_contextual_topk2_support_quality_pregate_pilot,
)


class ContextualTopk2SupportQualityPregatePilotTest(unittest.TestCase):
    def test_records_route_only_pilot_and_blocks_gpu_when_candidate_collapses_to_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            runpod = root / "runpod"
            active = root / "active.json"
            selector = root / "selector.json"
            review = root / "latest-review.md"
            _write_support_audit(local, linear_regret=0.02, linear_churn=0.25)
            _write_support_audit(runpod, linear_regret=0.03, linear_churn=0.24)
            _write_json(
                active,
                {
                    "status": "pass",
                    "decision": "causal_retention_claim_blocked_by_deployable_gate",
                    "claim_status": "local_retention_bracket_with_context_gated_singleton_efficacy_only",
                    "evidence": {
                        "metrics": {"deployable_gain_minus_ungated": -0.04},
                        "source_signals": {"retention_branch_supported": True},
                    },
                    "signals": {"local_retention_churn_bracket_supported": True},
                },
            )
            _write_json(
                selector,
                {
                    "status": "pass",
                    "decision": "post_active_topk1_contextual_topk2_branch_selected",
                    "selected_next_action": "design_support_quality_preserving_contextual_topk2_pregate",
                    "claim_status": "topk2_main_loop_selected_for_local_support_quality_redesign",
                },
            )
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        (
                            "recommended_next_action: Implement one executable local "
                            "support-quality-preserving contextual top-k-2 route-only pilot "
                            "with oracle-regret/one-swap training, hysteresis, same-student "
                            "support interventions, and linear/top-k-1/token-position/shuffled "
                            "controls before any GPU."
                        ),
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_contextual_topk2_support_quality_pregate_pilot(
                local_support_audit_dir=local,
                runpod_support_audit_dir=runpod,
                active_topk1_synthesis_path=active,
                branch_selector_path=selector,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], PILOT_RECORDED)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertEqual(
                summary["claim_status"],
                "route_only_contextual_topk2_support_quality_gate_failed_no_gpu",
            )
            self.assertIn(
                "candidate_reduces_mean_oracle_regret_vs_linear",
                {
                    row["criterion"]
                    for row in summary["gate_criteria"]
                    if not row["passed"]
                },
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "arm_metrics.csv").is_file())
            self.assertTrue((root / "report" / "fold_policy_rows.csv").is_file())
            self.assertTrue((root / "report" / "per_token_policy_rows.csv").is_file())
            with (root / "report" / "arm_metrics.csv").open(newline="", encoding="utf-8") as handle:
                arms = {row["arm"] for row in csv.DictReader(handle)}
            self.assertIn("per_token_one_swap_route_only", arms)
            self.assertIn("churn_aware_per_token_one_swap_route_only", arms)
            self.assertIn(
                "per_token_one_swap_does_not_increase_support_churn_vs_linear",
                {
                    row["criterion"]
                    for row in summary["gate_criteria"]
                    if not row["passed"]
                },
            )
            self.assertIn(
                "churn_aware_per_token_one_swap_does_not_increase_support_churn_vs_linear",
                {row["criterion"] for row in summary["gate_criteria"]},
            )

    def test_fails_closed_when_support_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = root / "local"
            active = root / "active.json"
            selector = root / "selector.json"
            review = root / "latest-review.md"
            _write_support_audit(local, linear_regret=0.02, linear_churn=0.25)
            _write_json(active, {"status": "pass"})
            _write_json(
                selector,
                {
                    "status": "pass",
                    "selected_next_action": "design_support_quality_preserving_contextual_topk2_pregate",
                },
            )
            review.write_text(
                "recommended_next_action: Implement one executable local support-quality-preserving contextual top-k-2 route-only pilot before any GPU.\n",
                encoding="utf-8",
            )

            summary = run_contextual_topk2_support_quality_pregate_pilot(
                local_support_audit_dir=local,
                runpod_support_audit_dir=root / "missing-runpod",
                active_topk1_synthesis_path=active,
                branch_selector_path=selector,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            self.assertIn(
                ("runpod_support_audit", "source_artifact"),
                {
                    (failure.get("source"), failure.get("field"))
                    for failure in summary["failures"]
                },
            )


def _write_support_audit(path: Path, *, linear_regret: float, linear_churn: float) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary.json",
        {
            "status": "pass",
            "decision": "causal_contextual_router_support_audit_blocks_promotion",
            "claim_status": "causal_contextual_router_ce_supported_support_quality_not_established",
        },
    )
    header = (
        "control,fold,heldout_sequence_index,router_loss,oracle_loss,"
        "oracle_support_regret,functional_churn_logit_l1,unique_support_sets,"
        "used_columns,support_change_fraction,shuffled_support_loss,random_support_loss,"
        "dominant_fixed_support_loss"
    )
    lines = [header]
    for fold in range(2):
        lines.append(
            (
                "causal_contextual_topk2,{fold},{fold},2.9,2.8,0.08,0.36,"
                "40,20,1.0,4.08,4.18,4.02"
            ).format(fold=fold)
        )
        lines.append(
            (
                "linear_topk2,{fold},{fold},3.5,3.48,{regret},{churn},"
                "15,10,0.9,4.1,4.16,4.0"
            ).format(fold=fold, regret=linear_regret, churn=linear_churn)
        )
        lines.append(
            (
                "full_context_oracle_topk2,{fold},{fold},2.84,2.83,0.01,0.37,"
                "35,18,1.0,4.2,4.21,4.01"
            ).format(fold=fold)
        )
    (path / "fold_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "aggregate_metrics.csv").write_text("control,mean_router_loss\n", encoding="utf-8")
    token_fieldnames = [
        "actual_support",
        "actual_support_loss",
        "best_one_swap_gain_vs_actual",
        "best_one_swap_improves_actual",
        "best_one_swap_regret",
        "best_one_swap_support",
        "best_one_swap_support_loss",
        "control",
        "flat_position",
        "fold",
        "one_swap_label_is_oracle",
        "oracle_support",
        "oracle_support_exact_match",
        "oracle_support_loss",
        "oracle_support_regret",
        "position_index",
        "router_support_loss",
        "sequence_index",
        "target_token",
    ]
    token_rows = []
    for fold in range(2):
        token_rows.extend(
            [
                _token_row(fold, 0, "linear_topk2", "0,1", "1,2", "1,2", 3.60, 3.50, 3.49, 0.11, True, 0.01),
                _token_row(fold, 1, "linear_topk2", "0,2", "2,3", "2,3", 3.62, 3.54, 3.52, 0.10, True, 0.02),
                _token_row(fold, 0, "causal_contextual_topk2", "3,4", "3,4", "1,2", 2.90, 2.90, 2.80, 0.10, False, 0.10),
                _token_row(fold, 1, "causal_contextual_topk2", "3,5", "3,5", "2,3", 2.95, 2.95, 2.82, 0.13, False, 0.13),
                _token_row(fold, 0, "full_context_oracle_topk2", "1,2", "1,2", "1,2", 2.80, 2.80, 2.80, 0.0, False, 0.0),
                _token_row(fold, 1, "full_context_oracle_topk2", "2,3", "2,3", "2,3", 2.82, 2.82, 2.82, 0.0, False, 0.0),
            ]
        )
    with (path / "per_token_support_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_fieldnames)
        writer.writeheader()
        writer.writerows(token_rows)


def _token_row(
    fold: int,
    position: int,
    control: str,
    actual_support: str,
    best_one_swap_support: str,
    oracle_support: str,
    actual_loss: float,
    best_one_swap_loss: float,
    oracle_loss: float,
    oracle_regret: float,
    improves: bool,
    best_one_swap_regret: float,
) -> dict[str, object]:
    return {
        "actual_support": actual_support,
        "actual_support_loss": actual_loss,
        "best_one_swap_gain_vs_actual": actual_loss - best_one_swap_loss,
        "best_one_swap_improves_actual": str(improves),
        "best_one_swap_regret": best_one_swap_regret,
        "best_one_swap_support": best_one_swap_support,
        "best_one_swap_support_loss": best_one_swap_loss,
        "control": control,
        "flat_position": position,
        "fold": fold,
        "one_swap_label_is_oracle": str(best_one_swap_support == oracle_support),
        "oracle_support": oracle_support,
        "oracle_support_exact_match": str(actual_support == oracle_support),
        "oracle_support_loss": oracle_loss,
        "oracle_support_regret": oracle_regret,
        "position_index": position,
        "router_support_loss": actual_loss,
        "sequence_index": fold,
        "target_token": 10 + position,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
