from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_design import (
    DESIGN_RECORDED,
    INSUFFICIENT_EVIDENCE,
    SELECTED_NEXT_ACTION,
    run_anticipatory_contextual_support_routing_design,
)


class AnticipatoryContextualSupportRoutingDesignTest(unittest.TestCase):
    def test_records_acsr_design_from_configs_and_retention_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)

            summary = run_anticipatory_contextual_support_routing_design(
                out_dir=root / "report",
                design_doc_path=paths["doc"],
                pilot_config_path=paths["pilot_config"],
                causal_config_path=paths["causal_config"],
                retention_decision_path=paths["retention"],
                strategy_review_path=paths["review"],
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], DESIGN_RECORDED)
            self.assertEqual(summary["selected_next_action"], SELECTED_NEXT_ACTION)
            self.assertEqual(
                summary["claim_statuses"]["full_context_contextual_router"],
                "nondeployable_oracle_teacher_only",
            )
            self.assertFalse(summary["strategy_review"]["ben_notification_required"])
            self.assertEqual(
                [target["name"] for target in summary["feature_targets"]],
                ["future_hidden", "future_delta"],
            )
            self.assertIn("current_hidden", summary["causal_inputs"])
            self.assertEqual(summary["pilot"]["support_router_teacher"], "contextual_mlp")
            self.assertEqual(summary["pilot"]["causal_control_router"], "contextual_mlp_causal")
            self.assertEqual(len(summary["control_rows"]), 8)
            self.assertEqual(len(summary["criteria_rows"]), 6)
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "source_rows.csv").is_file())
            self.assertTrue((root / "report" / "control_rows.csv").is_file())
            self.assertTrue((root / "report" / "criteria_rows.csv").is_file())
            self.assertTrue((root / "report" / "implementation_rows.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_without_retention_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            paths["retention"].unlink()

            summary = run_anticipatory_contextual_support_routing_design(
                out_dir=root / "report",
                design_doc_path=paths["doc"],
                pilot_config_path=paths["pilot_config"],
                causal_config_path=paths["causal_config"],
                retention_decision_path=paths["retention"],
                strategy_review_path=paths["review"],
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failures = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("retention_churn_decision", "source_artifact"), failures)
            self.assertIn(("retention_churn_decision", "status"), failures)


def _write_sources(root: Path) -> dict[str, Path]:
    paths = {
        "doc": root / "anticipatory_contextual_support_routing.md",
        "pilot_config": root / "pilot.yaml",
        "causal_config": root / "causal.yaml",
        "retention": root / "retention.json",
        "review": root / "latest-review.md",
    }
    paths["doc"].write_text(
        "\n".join(
            [
                "# Anticipatory Contextual Support Routing",
                "The branch uses anticipatory contextual support routing.",
                "It requires a shuffled predicted-feature control.",
                "It requires retention/churn evaluation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_config(paths["pilot_config"], support_router="contextual_mlp")
    _write_config(paths["causal_config"], support_router="contextual_mlp_causal")
    paths["retention"].write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "diagnose_retention_churn_microtest",
                "colab_replication_warranted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["review"].write_text(
        "\n".join(
            [
                "strategic_change_level: minor",
                "notify_ben: false",
                "recommended_next_action: Run local ACSR design",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _write_config(path: Path, *, support_router: str) -> None:
    path.write_text(
        f"""
run:
  experiment_id: test_{support_router}
  seed: 1
  max_steps: 50

data:
  dataset: tiny_shakespeare_word
  seq_len: 64

model:
  base:
    layers: 2
    hidden_dim: 96
  columns:
    num_columns: 24
    atoms_per_column: 4
    top_k: 2
    support_router: {support_router}
    support_stress_preset: false
""".strip()
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
