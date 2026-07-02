from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_order_averaging_closeout import (
    CLOSE_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_promoted_topk2_order_averaging_closeout,
)


class PromotedTopk2OrderAveragingCloseoutTest(unittest.TestCase):
    def test_closes_order_averaging_without_gpu_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)

            summary = run_promoted_topk2_order_averaging_closeout(
                order_averaging_probe_path=paths["probe"],
                inventory_path=paths["inventory"],
                multisite_closeout_path=paths["multisite"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "promoted_topk2_order_averaging_closed_no_gpu")
            self.assertEqual(summary["selected_next_action"], CLOSE_ACTION)
            self.assertEqual(
                summary["claim_status"],
                "order_averaging_closed_selector_required_for_next_deployable_mechanism",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            closeout = {row["branch"]: row for row in summary["closeout_rows"]}
            self.assertEqual(
                closeout["explicit_forward_reverse_order_averaging"]["disposition"],
                "closed_as_nondeployable_diagnostic",
            )
            self.assertEqual(closeout["gpu_validation"]["disposition"], "blocked")
            selected = [
                row for row in summary["candidate_actions"] if row["disposition"] == "selected"
            ]
            self.assertEqual(len(selected), 1)
            with (root / "out" / "candidate_actions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn(CLOSE_ACTION, {row["candidate_action"] for row in rows})
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("diagnostic", notes)
            self.assertIn("GPU validation remains blocked", notes)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_probe_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)
            paths["probe"].unlink()

            summary = run_promoted_topk2_order_averaging_closeout(
                order_averaging_probe_path=paths["probe"],
                inventory_path=paths["inventory"],
                multisite_closeout_path=paths["multisite"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            failure_fields = {
                (failure.get("source"), failure.get("field")) for failure in summary["failures"]
            }
            self.assertIn(
                ("explicit_order_averaging_mitigation_probe", "source_artifact"),
                failure_fields,
            )


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "probe": root / "probe.json",
        "inventory": root / "inventory.json",
        "multisite": root / "multisite.json",
        "review": root / "latest-review.md",
    }
    _write_json(
        paths["probe"],
        {
            "status": "pass",
            "decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
            "selected_next_action": "record_order_averaging_matched_control_closeout_no_gpu",
            "requires_gpu_now": False,
            "advance_to_gpu_validation": False,
            "promotion_allowed": False,
            "evidence": {
                "topk2_order_averaged_to_commutator_anchor_logit_mse_ratio": 0.25,
                "topk2_mean_order_averaged_anchor_ce_delta_vs_best_order": -0.05,
            },
            "gate_rows": [
                {
                    "gate": "flat_value_order_averaging_control_present",
                    "passes": False,
                },
                {
                    "gate": "promotion_or_gpu_allowed",
                    "passes": False,
                },
            ],
        },
    )
    _write_json(
        paths["inventory"],
        {
            "status": "pass",
            "decision": "commutator_dense_teacher_source_inventory_recorded",
            "claim_status": "commutator_inventory_selects_order_averaging_probe_no_gpu",
            "selected_next_action": "run_explicit_order_averaging_mitigation_probe_locally",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    )
    _write_json(
        paths["multisite"],
        {
            "status": "pass",
            "decision": "multisite_pc_core_periphery_branch_closed",
            "claim_status": "multisite_pc_core_periphery_closed_no_gpu",
            "selected_next_action": "close_multisite_pc_core_periphery_branch_before_gpu",
            "requires_gpu_now": False,
            "promotion_allowed": False,
        },
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: none",
                "notify_ben: false",
                "recommended_next_action: Patch launcher fallback and run local order averaging.",
                "verdict: PAUSE-RECOVER",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
