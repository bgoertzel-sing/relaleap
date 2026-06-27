from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.acsr_finite_update_commutator_assay import (
    REQUIRED_ARTIFACTS,
    _assay_gate_rows,
    _summary,
    _write_artifacts,
    run_acsr_finite_update_commutator_assay,
)


class ACSRFiniteUpdateCommutatorAssayTest(unittest.TestCase):
    def test_missing_config_fails_closed_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            summary = run_acsr_finite_update_commutator_assay(
                config_path=root / "missing.yaml",
                out_dir=root / "out",
                phase_steps=1,
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], "acsr_finite_update_commutator_assay_failed_closed")
            self.assertTrue(any(row["criterion"] == "config_present" for row in summary["failures"]))
            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((root / "out" / artifact).is_file(), artifact)

    def test_gate_passes_with_required_sparse_and_dense_commutator_rows(self) -> None:
        variant_rows = [
            _variant("sparse_acsr_contextual_topk2", "sparse", logit_mse=0.03, ce_abs=0.01),
            _variant("sparse_acsr_rank_matched_topk1", "sparse", logit_mse=0.02, ce_abs=0.01),
            _variant("dense_causal_rank1", "dense", logit_mse=0.05, ce_abs=0.02),
            _variant("dense_token_position_rank1", "dense", logit_mse=0.01, ce_abs=0.005),
        ]
        per_token_rows = [
            _per_token("sparse_acsr_contextual_topk2", "sparse"),
            _per_token("sparse_acsr_rank_matched_topk1", "sparse"),
            _per_token("dense_causal_rank1", "dense"),
            _per_token("dense_token_position_rank1", "dense"),
        ]

        gate_rows = _assay_gate_rows(
            variant_rows,
            per_token_rows,
            ce_guardrail=0.05,
            material_logit_mse_threshold=0.01,
        )

        self.assertTrue(all(row["passed"] for row in gate_rows), gate_rows)

    def test_gate_fails_when_dense_control_missing(self) -> None:
        variant_rows = [
            _variant("sparse_acsr_contextual_topk2", "sparse", logit_mse=0.03, ce_abs=0.01),
            _variant("sparse_acsr_rank_matched_topk1", "sparse", logit_mse=0.02, ce_abs=0.01),
        ]
        per_token_rows = [_per_token("sparse_acsr_contextual_topk2", "sparse")]

        gate_rows = _assay_gate_rows(
            variant_rows,
            per_token_rows,
            ce_guardrail=0.05,
            material_logit_mse_threshold=0.01,
        )

        self.assertTrue(
            any(row["criterion"] == "required_variants_present" and not row["passed"] for row in gate_rows)
        )
        self.assertTrue(
            any(row["criterion"] == "dense_control_available" and not row["passed"] for row in gate_rows)
        )

    def test_write_artifacts_for_synthetic_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out = root / "out"
            variant_rows = [
                _variant("sparse_acsr_contextual_topk2", "sparse", logit_mse=0.03, ce_abs=0.01),
                _variant("sparse_acsr_rank_matched_topk1", "sparse", logit_mse=0.02, ce_abs=0.01),
                _variant("dense_causal_rank1", "dense", logit_mse=0.05, ce_abs=0.02),
                _variant("dense_token_position_rank1", "dense", logit_mse=0.01, ce_abs=0.005),
            ]
            per_token_rows = [_per_token(row["variant"], row["family"]) for row in variant_rows]
            gate_rows = _assay_gate_rows(
                variant_rows,
                per_token_rows,
                ce_guardrail=0.05,
                material_logit_mse_threshold=0.01,
            )
            summary = _summary(
                status="pass",
                decision="acsr_sparse_commutator_lower_than_dense_control",
                start=0.0,
                config_path=root / "config.yaml",
                out_dir=out,
                phase_steps=1,
                variant_rows=variant_rows,
                per_token_rows=per_token_rows,
                strata_rows=[],
                gate_rows=gate_rows,
                ce_guardrail=0.05,
                material_logit_mse_threshold=0.01,
            )

            _write_artifacts(out, summary, variant_rows, per_token_rows, [], gate_rows)

            for artifact in REQUIRED_ARTIFACTS:
                self.assertTrue((out / artifact).is_file(), artifact)
            self.assertEqual(summary["metrics"]["sparse_minus_dense_logit_mse"], -0.020000000000000004)


def _variant(variant: str, family: str, *, logit_mse: float, ce_abs: float) -> dict[str, object]:
    return {
        "variant": variant,
        "family": family,
        "row_count": 1,
        "mean_logit_mse": logit_mse,
        "mean_symmetric_kl": logit_mse / 2.0,
        "mean_ce_abs_delta": ce_abs,
        "support_churn_fraction": 0.5 if family == "sparse" else None,
    }


def _per_token(variant: str, family: str) -> dict[str, object]:
    return {
        "variant": variant,
        "family": family,
        "split": "anchor",
        "batch_index": 0,
        "position_index": 0,
        "target_token": 7,
        "forward_ce": 1.0,
        "reverse_ce": 1.01,
        "ce_delta_forward_minus_reverse": -0.01,
        "ce_abs_delta": 0.01,
        "symmetric_kl": 0.01,
        "logit_mse": 0.02,
        "residual_delta_l2": 0.3,
        "forward_residual_l2": 0.4,
        "reverse_residual_l2": 0.5,
        "support_churn": "" if family == "dense" else "True",
        "forward_support": "" if family == "dense" else "1,2",
        "reverse_support": "" if family == "dense" else "2,3",
    }


if __name__ == "__main__":
    unittest.main()
