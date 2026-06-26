from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_retention_functional_churn_followup_report import (
    INSUFFICIENT_EVIDENCE,
    RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED,
    run_active_topk1_retention_functional_churn_followup_report,
)


class ActiveTopk1RetentionFunctionalChurnFollowupReportTest(unittest.TestCase):
    def test_report_supports_four_control_retention_churn_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection = root / "selection"
            seed1 = root / "seed1"
            seed2 = root / "seed2"
            _write_selection(selection)
            _write_probe(seed1, topk1_churn=0.004, topk2_churn=0.91)
            _write_probe(seed2, topk1_churn=0.008, topk2_churn=0.81)

            summary = run_active_topk1_retention_functional_churn_followup_report(
                selection_dir=selection,
                probe_dirs=(seed1, seed2),
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                RETENTION_FUNCTIONAL_CHURN_BRACKET_SUPPORTED,
            )
            self.assertTrue(summary["signals"]["required_four_controls_present"])
            self.assertTrue(
                summary["signals"]["topk1_transfer_beats_dense_and_random_controls"]
            )
            self.assertTrue(
                summary["signals"]["topk1_finite_update_commutator_cleaner_than_controls"]
            )
            self.assertGreater(
                summary["aggregates"][
                    "min_transfer_advantage_topk1_vs_random_fixed_topk2"
                ],
                0.0,
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue(
                (root / "report" / "packet_variant_metrics.csv").is_file()
            )
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_report_fails_closed_without_random_fixed_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection = root / "selection"
            seed1 = root / "seed1"
            _write_selection(selection)
            _write_probe(seed1, include_random=False)

            summary = run_active_topk1_retention_functional_churn_followup_report(
                selection_dir=selection,
                probe_dirs=(seed1,),
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("random_fixed_topk2.present", fields)


def _write_selection(out_dir: Path) -> None:
    out_dir.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "active_topk1_next_evidence_selected",
        "selected_experiment": "retention_churn",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_probe(
    out_dir: Path,
    *,
    include_random: bool = True,
    topk1_churn: float = 0.004,
    topk2_churn: float = 0.9,
) -> None:
    out_dir.mkdir(parents=True)
    variants = [
        _variant(
            "promoted_contextual_topk2",
            churn=topk2_churn,
            logit=0.16,
            transfer=0.91,
            commutator=0.21,
            ce_drift=0.01,
        ),
        _variant(
            "rank_matched_contextual_topk1",
            churn=topk1_churn,
            logit=0.14,
            transfer=0.95,
            commutator=0.04,
            ce_drift=0.01,
        ),
        _variant(
            "norm_matched_dense_active_rank",
            churn="",
            logit=0.05,
            transfer=0.42,
            commutator=0.08,
            ce_drift=0.02,
        ),
    ]
    if include_random:
        variants.append(
            _variant(
                "random_fixed_topk2",
                churn=0.0,
                logit=0.22,
                transfer=0.31,
                commutator=0.30,
                ce_drift=0.03,
            )
        )
    summary = {
        "status": "pass",
        "decision": "active_topk1_retention_churn_probe_established",
        "config_path": "configs/test.yaml",
        "audit": {"variants": variants},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _variant(
    name: str,
    *,
    churn: float | str,
    logit: float,
    transfer: float,
    commutator: float,
    ce_drift: float,
) -> dict[str, object]:
    return {
        "variant": name,
        "anchor_ce_drift": ce_drift,
        "anchor_logit_mse_drift": logit,
        "anchor_residual_stream_l2_drift": 1.0,
        "anchor_support_churn_after_transfer": churn,
        "transfer_ce_improvement": transfer,
        "commutator_anchor_logit_mse": commutator,
        "commutator_transfer_logit_mse": commutator,
        "commutator_anchor_residual_stream_l2": 1.0,
        "commutator_transfer_residual_stream_l2": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
