from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_finite_update_order_control_audit import (
    FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
)
from relaleap.experiments.promoted_topk2_pairwise_value_interaction_localization_audit import (
    PAIRWISE_VALUE_INTERACTION_DIFFUSE,
)
from relaleap.experiments.promoted_topk2_post_finite_update_closeout_report import (
    INSUFFICIENT_EVIDENCE,
    POST_FINITE_UPDATE_CLOSEOUT_SELECTED,
    SELECTED_NEXT_ACTION,
    run_promoted_topk2_post_finite_update_closeout_report,
)
from relaleap.experiments.promoted_topk2_retention_synthesis_gate import (
    CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
)


class PromotedTopk2PostFiniteUpdateCloseoutReportTest(unittest.TestCase):
    def test_selects_control_matrix_extension_after_diffuse_pairwise_closeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Rerun pairwise localization before new mitigation",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_promoted_topk2_post_finite_update_closeout_report(
                pairwise_localization_path=paths["pairwise"],
                finite_update_report_path=paths["finite_update"],
                retention_synthesis_path=paths["retention"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POST_FINITE_UPDATE_CLOSEOUT_SELECTED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertEqual(
                summary["claim_statuses"]["value_router_mitigation_family"],
                "closed_for_now_diffuse_or_not_established",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertIn("per-token forward-vs-reverse CE", summary["next_step"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "selected_next_step.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_finite_update_report_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["finite_update"].unlink()

            summary = run_promoted_topk2_post_finite_update_closeout_report(
                pairwise_localization_path=paths["pairwise"],
                finite_update_report_path=paths["finite_update"],
                retention_synthesis_path=paths["retention"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("finite_update_order_control", "source_artifact"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "pairwise": root / "pairwise.json",
        "finite_update": root / "finite_update.json",
        "retention": root / "retention.json",
    }
    _write_json(
        paths["pairwise"],
        {
            "status": "pass",
            "decision": PAIRWISE_VALUE_INTERACTION_DIFFUSE,
            "localization_status": "diffuse",
            "metrics": {
                "top1_pair_abs_mass_fraction": 0.14,
                "discovery_confirmation_top3_overlap": 0.33,
            },
        },
    )
    _write_json(
        paths["finite_update"],
        {
            "status": "pass",
            "decision": FINITE_UPDATE_ORDER_SENSITIVITY_CE_BOUNDED,
            "metrics": {
                "topk2_mean_commutator_anchor_ce_abs_delta": 0.015,
                "topk2_mean_commutator_anchor_logit_mse": 0.24,
                "topk2_to_topk1_mean_commutator_anchor_logit_mse_ratio": 25.8,
                "topk2_to_dense_mean_commutator_anchor_logit_mse_ratio": 3.4,
                "per_token_commutator_row_count": 4032,
                "per_token_commutator_ce_abs_delta_mean": 0.25,
                "per_token_commutator_symmetric_kl_mean": 0.13,
            },
        },
    )
    _write_json(
        paths["retention"],
        {
            "status": "pass",
            "decision": CONTEXTUAL_TOPK2_ROUTER_DEFAULT_TOPK1_DIAGNOSTIC,
            "next_step": "run a local no-training finite-update order-symmetrization audit",
        },
    )
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
