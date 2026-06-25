from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_post_bracket_research_direction_report import (
    BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED,
    INSUFFICIENT_EVIDENCE,
    POST_BRACKET_DIRECTION_SELECTED,
    SELECTED_EXPERIMENT,
    run_active_topk1_post_bracket_research_direction_report,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


class ActiveTopk1PostBracketResearchDirectionReportTest(unittest.TestCase):
    def test_report_selects_context_conditioned_interference_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: run context-conditioned singleton interference decomposition",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_active_topk1_post_bracket_research_direction_report(
                backend_provenance_dir=paths["backend"],
                functional_retention_dir=paths["retention"],
                singleton_reconciliation_dir=paths["singleton"],
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], POST_BRACKET_DIRECTION_SELECTED)
            self.assertEqual(summary["selected_experiment"], SELECTED_EXPERIMENT)
            self.assertEqual(
                summary["claim_policy"], BROAD_REUSABLE_SINGLETON_CLAIM_EXCLUDED
            )
            self.assertFalse(summary["experiment_design"]["requires_gpu_now"])
            self.assertIn(
                "off_context_forced_singleton_matched",
                {
                    row["component"]
                    for row in summary["experiment_design"]["components"]
                },
            )
            self.assertIn(
                "support_jaccard_distance", summary["experiment_design"]["estimands"]
            )
            self.assertEqual(summary["strategy_review"]["strategic_change_level"], "minor")
            self.assertFalse(summary["strategy_review"]["notify_ben"])
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "selected_experiment.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_report_fails_closed_when_singleton_reconciliation_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = _write_sources(root)
            (paths["singleton"] / "summary.json").unlink()

            summary = run_active_topk1_post_bracket_research_direction_report(
                backend_provenance_dir=paths["backend"],
                functional_retention_dir=paths["retention"],
                singleton_reconciliation_dir=paths["singleton"],
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(("singleton_reconciliation_audit", "summary_json"), fields)


def _write_sources(root: Path) -> dict[str, Path]:
    backend = root / "backend"
    retention = root / "retention"
    singleton = root / "singleton"
    backend.mkdir()
    retention.mkdir()
    singleton.mkdir()
    _write_json(
        backend / "summary.json",
        {
            "status": "pass",
            "decision": ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
        },
    )
    _write_json(
        retention / "summary.json",
        {
            "status": "pass",
            "decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
            "claim_status": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        },
    )
    _write_json(
        singleton / "summary.json",
        {
            "status": "pass",
            "decision": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        },
    )
    return {"backend": backend, "retention": retention, "singleton": singleton}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
