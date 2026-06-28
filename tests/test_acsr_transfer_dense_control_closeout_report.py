from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_transfer_dense_control_closeout_report import (
    REQUIRED_ARTIFACTS,
    run_acsr_transfer_dense_control_closeout_report,
)


class ACSRTransferDenseControlCloseoutReportTest(unittest.TestCase):
    def test_closeout_records_dense_blocker_and_ben_notify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation = root / "validation.json"
            heldout = root / "heldout.json"
            dense = root / "dense.json"
            review = root / "latest-review.md"
            _write_validation(validation)
            _write_heldout(heldout)
            _write_dense(dense)
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: major",
                        "notify_ben: true",
                        "recommended_next_action: Pivot to ACSR locally, no GPU",
                        "verdict: PIVOT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_transfer_dense_control_closeout_report(
                transfer_validation_path=validation,
                heldout_control_path=heldout,
                dense_transfer_path=dense,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["claim_status"],
                "transfer_objective_not_separated_from_dense_controls",
            )
            self.assertFalse(summary["requires_gpu_now"])
            self.assertTrue(summary["direction_shift"]["ben_should_be_notified"])
            self.assertIn("stop ACSR transfer-objective promotion", summary["selected_next_step"])
            self.assertTrue(
                any(
                    row["criterion"] == "dense_transfer_control_available"
                    and row["passed"]
                    for row in summary["closeout_criteria"]
                )
            )
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_missing_dense_control_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation = root / "validation.json"
            heldout = root / "heldout.json"
            _write_validation(validation)
            _write_heldout(heldout)

            summary = run_acsr_transfer_dense_control_closeout_report(
                transfer_validation_path=validation,
                heldout_control_path=heldout,
                dense_transfer_path=root / "missing.json",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(
                summary["decision"],
                "acsr_transfer_dense_control_closeout_failed_closed",
            )
            self.assertTrue(summary["failures"])


def _write_validation(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "claim_status": "cross_backend_transfer_objective_supported_not_promoted",
            "aggregate_metrics": {
                "mean_partner_transfer_minus_direct_ce": -0.03,
            },
        },
    )


def _write_heldout(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "pass",
            "claim_status": "heldout_transfer_controls_supported_not_promoted",
            "aggregate_metrics": {
                "mean_heldout_partner_transfer_minus_direct_ce": -0.02,
                "max_heldout_own_transfer_minus_direct_ce": 0.01,
            },
        },
    )


def _write_dense(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "fail",
            "claim_status": "sparse_transfer_not_separated_from_dense_control",
            "failures": [{"criterion": "sparse_transfer_beats_causal_dense_control"}],
            "source_metrics": [
                {
                    "value_path": "partner_values",
                    "arm": "transfer_objective_router",
                    "heldout_delta_vs_direct_ce": -0.03,
                }
            ],
            "dense_control_rows": [
                {
                    "control": "rank_matched_causal_dense_residual",
                    "heldout_delta_vs_base_ce": -0.26,
                }
            ],
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
