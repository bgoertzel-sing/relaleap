from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.deployable_commutator_regularized_sparse_update_pregate import (
    IMPLEMENT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_deployable_commutator_regularized_sparse_update_pregate,
)


class DeployableCommutatorRegularizedSparseUpdatePregateTests(unittest.TestCase):
    def test_records_local_pregate_from_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)

            summary = run_deployable_commutator_regularized_sparse_update_pregate(
                selector_path=paths["selector"],
                order_probe_path=paths["order"],
                value_penalty_path=paths["value"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                "deployable_commutator_regularized_sparse_update_pregate_recorded",
            )
            self.assertEqual(summary["selected_next_action"], IMPLEMENT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            arms = {row["arm"] for row in summary["control_arms"]}
            self.assertIn("dense_active_matched_update", arms)
            self.assertIn("same_router_flat_value_update", arms)
            self.assertIn("random_support_sparse_update", arms)
            self.assertIn("no_update", arms)
            components = {row["component"] for row in summary["update_contract"]}
            self.assertIn("candidate_update_rule", components)
            self.assertIn("nondeployable_upper_bound", components)
            metrics = {row["metric"] for row in summary["observable_gates"]}
            self.assertIn("parameter_commutator_norm", metrics)
            self.assertIn("behavioral_logit_commutator_kl", metrics)
            self.assertIn("old_task_forgetting", metrics)
            self.assertTrue(all(row["passed"] for row in summary["gate_rows"]))
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("GPU validation remains blocked", notes)
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_selector_fails_closed_but_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_sources(root)
            paths["selector"].unlink()

            summary = run_deployable_commutator_regularized_sparse_update_pregate(
                selector_path=paths["selector"],
                order_probe_path=paths["order"],
                value_penalty_path=paths["value"],
                strategy_review_path=paths["review"],
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "selector": root / "selector.json",
        "order": root / "order.json",
        "value": root / "value.json",
        "review": root / "latest-review.md",
    }
    _write_json(
        paths["selector"],
        {
            "status": "pass",
            "decision": "post_order_averaging_deployable_mechanism_selected",
            "claim_status": "deployable_commutator_regularized_sparse_update_pregate_selected_no_gpu",
            "selected_next_action": "design_deployable_commutator_regularized_sparse_update_pregate",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
        },
    )
    _write_json(
        paths["order"],
        {
            "status": "pass",
            "decision": "explicit_order_averaging_diagnostic_candidate_not_promoted",
            "requires_gpu_now": False,
            "promotion_allowed": False,
            "advance_to_gpu_validation": False,
        },
    )
    _write_json(
        paths["value"],
        {
            "status": "pass",
            "decision": "commutator_value_penalty_not_established",
            "requires_gpu_now": False,
        },
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: none",
                "notify_ben: false",
                "recommended_next_action: run local commutator mechanism pregate",
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
