from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_value_dictionary_capacity_rescue_closeout import (
    NEXT_BRANCH_SELECTOR_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_value_dictionary_capacity_rescue_closeout,
)


class LowChurnMlpValueDictionaryCapacityRescueCloseoutTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_closeout(
                pregate_dir=root / "missing",
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_closes_target_aware_value_dictionary_rescue_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pregate = root / "pregate"
            _write_pregate(pregate)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Verify null and budget labels before closeout.",
                        "verdict: FIX",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_closeout(
                pregate_dir=pregate,
                strategy_review_path=review,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], "low_churn_mlp_value_dictionary_capacity_rescue_closed")
            self.assertEqual(summary["selected_next_action"], NEXT_BRANCH_SELECTOR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            selected = [row for row in summary["candidate_actions"] if row["disposition"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["candidate_action"], NEXT_BRANCH_SELECTOR_ACTION)
            signals = {row["signal"]: row["passed"] for row in summary["closeout_rows"] if row["required"]}
            self.assertTrue(signals["best_sparse_is_target_aware_nondeployable"])
            self.assertTrue(signals["valid_target_aware_null_ties_sparse"])
            self.assertTrue(signals["budget_matched_low_rank_dominates_sparse"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("did not rescue the branch", notes)


def _write_pregate(path: Path) -> None:
    path.mkdir(parents=True)
    sparse = {
        "candidate": "multi_codebook_residual_dictionary",
        "family": "sparse_oracle",
        "heldout_reconstruction_r2": 0.6452806146191838,
        "target_access_at_eval": "target_residual_vector",
        "support_source": "heldout_target_nearest_code",
        "deployable": False,
        "valid_null_for_target_access": False,
    }
    null = {
        "candidate": "shuffled_teacher_dictionary",
        "family": "null",
        "heldout_reconstruction_r2": 0.6452806146191838,
        "target_access_at_eval": "target_residual_vector",
        "support_source": "heldout_target_nearest_code",
        "deployable": False,
        "valid_null_for_target_access": True,
    }
    control = {
        "candidate": "low_rank_svd_rank32",
        "family": "capacity_control",
        "heldout_reconstruction_r2": 1.0,
        "budget_match_group": "full_rank_ceiling",
    }
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_value_dictionary_capacity_rescue_pregate_recorded",
                "claim_status": "value_dictionary_capacity_rescue_local_gates_block_gpu",
                "selected_next_action": "close_value_dictionary_capacity_rescue_or_request_strategy_review",
                "requires_gpu_now": False,
                "promotion_allowed": False,
                "advance_to_gpu_validation": False,
                "best_sparse_oracle": sparse,
                "best_valid_null": null,
                "best_capacity_control": control,
                "best_sparse_oracle_r2": 0.6452806146191838,
                "valid_null_delta_r2": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        path / "candidate_metrics.csv",
        [
            sparse | {"active_value_count": 16, "budget_match_group": "sparse_value_budget"},
            null | {"active_value_count": 16, "budget_match_group": "sparse_value_budget"},
            {
                "candidate": "low_rank_svd_rank16",
                "family": "capacity_control",
                "heldout_reconstruction_r2": 0.999,
                "budget_match_group": "budget_matched_low_rank",
            },
            control,
        ],
    )
    _write_csv(
        path / "gate_criteria.csv",
        [
            {"criterion": "richer_sparse_oracle_min_r2", "passed": False, "gate_type": "scientific_advancement"},
            {"criterion": "richer_sparse_improves_single_codebook", "passed": True, "gate_type": "scientific_advancement"},
            {"criterion": "dense_low_rank_control_not_dominant", "passed": False, "gate_type": "scientific_advancement"},
            {"criterion": "shuffled_and_route_nulls_rejected", "passed": False, "gate_type": "scientific_advancement"},
        ],
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
