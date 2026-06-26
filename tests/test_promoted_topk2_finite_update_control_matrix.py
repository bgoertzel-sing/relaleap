from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_finite_update_control_matrix import (
    FINITE_UPDATE_CONTROL_MATRIX_READY,
    INSUFFICIENT_EVIDENCE,
    run_promoted_topk2_finite_update_control_matrix,
)


class PromotedTopk2FiniteUpdateControlMatrixTest(unittest.TestCase):
    def test_builds_no_training_control_matrix_from_per_token_packets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "finite_report"
            packet = root / "packet"
            microtest = root / "microtest"
            _write_finite_report(report, [packet, microtest])
            _write_per_token(packet, include_random=False)
            _write_per_token(microtest, include_random=True)

            summary = run_promoted_topk2_finite_update_control_matrix(
                finite_update_report_dir=report,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], FINITE_UPDATE_CONTROL_MATRIX_READY)
            self.assertTrue(summary["signals"]["all_required_variants_present"])
            self.assertTrue(summary["signals"]["required_per_token_fields_present"])
            self.assertTrue(summary["signals"]["topk2_logit_mse_exceeds_topk1"])
            self.assertTrue(summary["signals"]["topk2_support_churn_high"])
            self.assertEqual(summary["metrics"]["topk2_row_count"], 4)
            self.assertEqual(summary["metrics"]["topk1_row_count"], 4)
            self.assertEqual(summary["metrics"]["random_fixed_topk2_row_count"], 2)
            self.assertEqual(summary["metrics"]["dense_active_rank_row_count"], 4)
            self.assertEqual(summary["claim_gate"], "matrix_input_only_not_causal_cooperation_evidence")
            self.assertIn("causal fingerprint/control audit", summary["next_step"])
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue(
                (root / "out" / "finite_update_control_matrix.csv").is_file()
            )
            self.assertTrue(
                (root / "out" / "finite_update_control_strata.csv").is_file()
            )
            with (root / "out" / "finite_update_control_strata.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                strata_rows = list(csv.DictReader(handle))
            self.assertTrue(
                {
                    "support_transition",
                    "matrix_role_x_position_bin",
                    "matrix_role_x_support_churn",
                    "matrix_role_x_residual_delta_l2_bin",
                }.issubset({row["stratum"] for row in strata_rows})
            )

    def test_fails_closed_when_required_control_variant_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "finite_report"
            packet = root / "packet"
            _write_finite_report(report, [packet])
            _write_per_token(packet, include_random=False, include_dense=False)

            summary = run_promoted_topk2_finite_update_control_matrix(
                finite_update_report_dir=report,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            missing = {
                failure["expected"]
                for failure in summary["failures"]
                if failure["field"] == "required_variant"
            }
            self.assertIn("random_fixed_topk2", missing)
            self.assertIn("norm_matched_dense_active_rank", missing)


def _write_finite_report(path: Path, packet_dirs: list[Path]) -> None:
    path.mkdir(parents=True)
    packet_rows = [
        {
            "packet": f"packet_{index}",
            "probe_dir": str(packet_dir),
            "status": "pass",
            "decision": "source-ready",
            "config_path": f"configs/source_{index}.yaml",
        }
        for index, packet_dir in enumerate(packet_dirs, start=1)
    ]
    (path / "summary.json").write_text(
        json.dumps({"packet_rows": packet_rows, "microtest_packet_rows": []}),
        encoding="utf-8",
    )


def _write_per_token(
    path: Path,
    *,
    include_random: bool,
    include_dense: bool = True,
) -> None:
    path.mkdir(parents=True)
    rows = []
    variants = [
        ("promoted_contextual_topk2", 0.4, 0.2, 5.0, "True", "1,2", "2,3"),
        ("rank_matched_contextual_topk1", 0.1, 0.01, 1.0, "False", "4", "4"),
    ]
    if include_random:
        variants.append(
            ("random_fixed_topk2", 0.5, 0.3, 6.0, "False", "0,5", "0,5")
        )
    if include_dense:
        variants.append(
            ("norm_matched_dense_active_rank", 0.08, 0.05, 3.0, "", "", "")
        )
    for split in ("anchor", "transfer"):
        for (
            variant,
            ce_abs,
            logit_mse,
            residual_delta,
            churn,
            forward_support,
            reverse_support,
        ) in variants:
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "batch_index": 0,
                    "position_index": 0,
                    "position_bin": "even",
                    "target_token": 7,
                    "token_class": "rare_target",
                    "forward_ce": 1.0,
                    "reverse_ce": 1.0 + ce_abs,
                    "ce_delta_forward_minus_reverse": -ce_abs,
                    "ce_abs_delta": ce_abs,
                    "symmetric_kl": logit_mse / 2.0,
                    "logit_mse": logit_mse,
                    "residual_delta_l2": residual_delta,
                    "residual_norm": 8.0,
                    "residual_norm_bin": "mid",
                    "residual_delta_l2_bin": "high",
                    "support_churn": churn,
                    "forward_support": forward_support,
                    "reverse_support": reverse_support,
                }
            )
    with (path / "per_token_commutator.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
