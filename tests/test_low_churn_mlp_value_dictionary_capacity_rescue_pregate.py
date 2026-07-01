from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.low_churn_mlp_value_dictionary_capacity_rescue_pregate import (
    NEXT_ACTION,
    REPAIR_ACTION,
    REQUIRED_ARTIFACTS,
    run_low_churn_mlp_value_dictionary_capacity_rescue_pregate,
)


class LowChurnMlpValueDictionaryCapacityRescuePregateTests(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_pregate(
                design_dir=root / "missing_design",
                vector_capture_dir=root / "missing_capture",
                decision_audit_dir=root / "missing_audit",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["selected_next_action"], REPAIR_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertTrue(summary["runtime_failures"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_records_richer_dictionary_pregate_and_keeps_gpu_blocked_without_incremental_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            design = root / "design"
            capture = root / "capture"
            audit = root / "audit"
            _write_design(design)
            _write_audit(audit)
            _write_capture(capture)

            summary = run_low_churn_mlp_value_dictionary_capacity_rescue_pregate(
                design_dir=design,
                vector_capture_dir=capture,
                decision_audit_dir=audit,
                out_dir=root / "out",
                dictionary_size=2,
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["selected_next_action"], NEXT_ACTION)
            self.assertFalse(summary["requires_gpu_now"])
            self.assertFalse(summary["promotion_allowed"])
            self.assertFalse(summary["advance_to_gpu_validation"])
            self.assertGreaterEqual(summary["best_sparse_oracle_r2"], 0.65)
            self.assertEqual(summary["valid_null_delta_r2"], 0.0)
            self.assertTrue(summary["advancement_failures"])

            with (root / "out" / "candidate_metrics.csv").open(newline="", encoding="utf-8") as handle:
                candidates = list(csv.DictReader(handle))
            names = {row["candidate"] for row in candidates}
            self.assertIn("multi_codebook_residual_dictionary", names)
            self.assertIn("low_rank_svd_rank3", names)
            self.assertIn("shuffled_teacher_dictionary", names)
            shuffled = [row for row in candidates if row["candidate"] == "shuffled_teacher_dictionary"][0]
            self.assertEqual(shuffled["target_access_at_eval"], "target_residual_vector")
            self.assertEqual(shuffled["valid_null_for_target_access"], "True")
            full_rank = [row for row in candidates if row["candidate"] == "low_rank_svd_rank3"][0]
            self.assertEqual(full_rank["budget_match_group"], "full_rank_ceiling")

            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("local ceilings, not deployable claims", notes)


def _write_design(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_value_dictionary_capacity_rescue_design_recorded",
                "selected_next_action": "implement_value_dictionary_capacity_rescue_local_pregate",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_audit(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "low_churn_mlp_sparse_factorization_decision_audit_recorded",
                "selected_next_action": "redesign_value_dictionary_or_close_sparse_factorization_ceiling",
                "advance_to_gpu_validation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_capture(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    for index in range(8):
        split = "heldout" if index >= 4 else "train_anchor"
        base = index % 4
        vector = [float(base % 2), float((base + 1) % 2), 0.0]
        rows.append(
            {
                "teacher_row_id": f"row{index}",
                "split": split,
                "token_index": index,
                "teacher_residual_update_vector": json.dumps(vector),
            }
        )
    _write_csv(path / "raw_teacher_residual_vectors.csv", rows)


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
