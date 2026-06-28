from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.dense_teacher_control_mechanism_assay import (
    REQUIRED_ARTIFACTS,
    run_dense_teacher_control_mechanism_assay,
)


class DenseTeacherControlMechanismAssayTest(unittest.TestCase):
    def test_missing_sources_fail_closed_but_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_dense_teacher_control_mechanism_assay(
                dense_teacher_dir=root / "missing_dense_teacher",
                dense_primary_dir=root / "missing_dense_primary",
                mlp_followup_dir=root / "missing_mlp_followup",
                mlp_fingerprint_dir=root / "missing_mlp_fingerprint",
                matched_decision_dir=root / "missing_matched",
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["scientific_gate"], "blocked")
            self.assertEqual(summary["decision"], "dense_teacher_control_mechanism_assay_blocked")
            self.assertFalse(summary["requires_gpu_now"])
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_blocks_when_dense_teacher_and_matched_gate_do_not_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dense_teacher = root / "dense_teacher"
            dense_primary = root / "dense_primary"
            mlp_followup = root / "mlp_followup"
            mlp_fingerprint = root / "mlp_fingerprint"
            matched = root / "matched"
            _write_dense_teacher(dense_teacher, status="fail")
            _write_dense_primary(dense_primary)
            _write_mlp_followup(mlp_followup)
            _write_mlp_fingerprint(mlp_fingerprint)
            _write_matched(matched, scientific_gate="blocked")

            summary = run_dense_teacher_control_mechanism_assay(
                dense_teacher_dir=dense_teacher,
                dense_primary_dir=dense_primary,
                mlp_followup_dir=mlp_followup,
                mlp_fingerprint_dir=mlp_fingerprint,
                matched_decision_dir=matched,
                out_dir=root / "out",
            )

            self.assertEqual(summary["scientific_gate"], "blocked")
            criteria = {row["criterion"]: row for row in summary["criteria"]}
            self.assertFalse(criteria["dense_teacher_acsr_pilot_supported"]["passed"])
            self.assertFalse(criteria["matched_dense_control_gate_passed"]["passed"])
            self.assertTrue(criteria["dense24_and_mlp_control_fields_present"]["passed"])
            notes = (root / "out" / "notes.md").read_text(encoding="utf-8")
            self.assertIn("dense24", notes)


def _write_dense_teacher(path: Path, *, status: str) -> None:
    path.mkdir(parents=True)
    decision = (
        "dense_teacher_residual_distillation_acsr_pilot_supported_not_promoted"
        if status == "pass"
        else "dense_teacher_residual_distillation_pilot_not_supported"
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "decision": decision,
                "variant_rows": [
                    {
                        "variant": "acsr_predicted_future_support",
                        "ce_loss": 2.8,
                        "teacher_logit_mse": 10.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dense_primary(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "dense_primary_mechanism_assay_selected",
                "candidate_scorecard": [
                    _scorecard_row("dense_rank24_best_norm", 3.8, 1.0, 0.02, 0.27, 1.0),
                    _scorecard_row("parameter_matched_causal_mlp_control", 2.9, 4.4, 0.15, 0.75, 1.0),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_mlp_followup(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "mlp_primary_with_functional_churn_tradeoff",
                "mechanism_comparison": [
                    _comparison_row("dense_rank24_best_norm", 3.76, 1.0, 0.014, 0.24),
                    _comparison_row("parameter_matched_causal_mlp_control", 2.87, 4.4, 0.154, 0.84),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_mlp_fingerprint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "mlp_churn_intervention_fingerprint_scaled_assay_completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_matched(path: Path, *, scientific_gate: str) -> None:
    path.mkdir(parents=True)
    domination = path / "domination_cases.csv"
    _write_csv(
        domination,
        [
            {"challenger_arm": "parameter_matched_causal_mlp_control", "challenger_advances": False},
            {"challenger_arm": "dense_rank24_best_norm", "challenger_advances": False},
        ],
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "matched_intervention_challengers_do_not_clear_best_dense_pareto_guardrail",
                "scientific_gate": scientific_gate,
                "advancement_row_count": 0,
                "artifacts": {"domination_cases_csv": str(domination)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _scorecard_row(
    arm: str,
    ce_loss: float,
    residual_l2: float,
    anchor: float,
    churn: float,
    purity: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "ce_loss": ce_loss,
        "residual_l2": residual_l2,
        "anchor_kl_or_logit_mse": anchor,
        "functional_churn": churn,
        "intervention_fingerprint_purity": purity,
    }


def _comparison_row(
    arm: str,
    ce_loss: float,
    residual_l2: float,
    logit_mse: float,
    flip: float,
) -> dict[str, object]:
    return {
        "arm": arm,
        "heldout_ce_loss": ce_loss,
        "heldout_residual_update_l2": residual_l2,
        "heldout_logit_mse_vs_base": logit_mse,
        "heldout_prediction_changed_vs_base": flip,
    }


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
