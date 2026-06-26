from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.promoted_topk2_finite_update_augmented_causal_gate import (
    BLOCKED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    run_promoted_topk2_finite_update_augmented_causal_gate,
)


class PromotedTopk2FiniteUpdateAugmentedCausalGateTest(unittest.TestCase):
    def test_supports_only_when_benefit_survives_risk_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            deconfounded = root / "deconfounded"
            finite = root / "finite"
            packet = root / "packet"
            _write_deconfounded(deconfounded, decision_supports=True)
            _write_finite_matrix(finite, packet)
            _write_per_token(packet, risky_topk2=False)

            summary = run_promoted_topk2_finite_update_augmented_causal_gate(
                deconfounded_dir=deconfounded,
                finite_update_matrix_dir=finite,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], SUPPORTED)
            self.assertTrue(summary["signals"]["benefit_fraction_gate_passed"])
            self.assertTrue(summary["signals"]["topk2_logit_mse_not_worse_than_topk1"])
            self.assertEqual(summary["metrics"]["augmented_strata_count"], 2)
            self.assertTrue((root / "out" / "summary.json").is_file())
            self.assertTrue(
                (root / "out" / "augmented_deconfounded_strata.csv").is_file()
            )
            self.assertTrue(
                (root / "out" / "finite_update_risk_controls.csv").is_file()
            )

    def test_blocks_when_finite_update_risk_is_worse_than_topk1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            deconfounded = root / "deconfounded"
            finite = root / "finite"
            packet = root / "packet"
            _write_deconfounded(deconfounded, decision_supports=True)
            _write_finite_matrix(finite, packet)
            _write_per_token(packet, risky_topk2=True)

            summary = run_promoted_topk2_finite_update_augmented_causal_gate(
                deconfounded_dir=deconfounded,
                finite_update_matrix_dir=finite,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], BLOCKED)
            self.assertFalse(
                summary["signals"]["topk2_logit_mse_not_worse_than_topk1"]
            )
            self.assertFalse(summary["signals"]["topk2_support_churn_not_high"])
            self.assertGreater(
                summary["metrics"]["augmented_mean_topk2_minus_topk1_finite_logit_mse"],
                0.0,
            )

    def test_fails_closed_when_required_control_role_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            deconfounded = root / "deconfounded"
            finite = root / "finite"
            packet = root / "packet"
            _write_deconfounded(deconfounded, decision_supports=True)
            _write_finite_matrix(finite, packet)
            _write_per_token(packet, risky_topk2=False, include_dense=False)

            summary = run_promoted_topk2_finite_update_augmented_causal_gate(
                deconfounded_dir=deconfounded,
                finite_update_matrix_dir=finite,
                out_dir=root / "out",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            missing = [
                failure
                for failure in summary["failures"]
                if failure["field"] == "finite_update_required_role"
            ]
            self.assertEqual(missing[0]["expected"], "dense_active_rank")


def _write_deconfounded(path: Path, *, decision_supports: bool) -> None:
    path.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": (
            "topk2_causal_metrics_survive_deconfounding_with_ce_guardrail"
            if decision_supports
            else "topk2_comparative_causal_cooperation_not_supported"
        ),
        "evidence": {
            "ce_guardrail_passed": True,
            "topk2_incremental_pair_gain_positive_strata_fraction": 1.0,
            "topk2_fixed_support_cleaner_strata_fraction": 1.0,
            "topk2_functional_churn_cleaner_strata_fraction": 1.0,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        _deconf_row("even", "common_target", "low"),
        _deconf_row("odd", "rare_target", "mid"),
    ]
    _write_csv(path / "matched_deconfounded_strata.csv", list(rows[0]), rows)


def _deconf_row(position_bin: str, token_class: str, residual_norm_bin: str) -> dict[str, object]:
    return {
        "position_bin": position_bin,
        "token_class": token_class,
        "residual_norm_bin": residual_norm_bin,
        "residual_gain_bin": "low",
        "support_count_bin": "low",
        "matched_exact_context_count": 3,
        "topk2_incremental_pair_gain_minus_topk1_singleton": 0.4,
        "topk2_fixed_delta_minus_topk1": -0.2,
        "topk2_logit_mse_minus_topk1": -0.1,
        "topk2_residual_stream_l2_delta_minus_topk1": -0.3,
    }


def _write_finite_matrix(path: Path, packet: Path) -> None:
    path.mkdir(parents=True)
    summary = {
        "status": "pass",
        "decision": "finite_update_control_matrix_ready",
        "metrics": {
            "topk2_minus_topk1_logit_mse": 0.2,
            "topk2_minus_dense_logit_mse": 0.1,
            "topk2_minus_random_fixed_topk2_logit_mse": -0.1,
            "topk2_support_churn_fraction": 0.9,
        },
    }
    (path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        path / "source_rows.csv",
        ["packet", "probe_dir", "per_token_commutator_present"],
        [
            {
                "packet": "packet_1",
                "probe_dir": str(packet),
                "per_token_commutator_present": "True",
            }
        ],
    )


def _write_per_token(
    path: Path,
    *,
    risky_topk2: bool,
    include_dense: bool = True,
) -> None:
    path.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    topk2_logit_mse = 0.5 if risky_topk2 else 0.05
    topk2_churn = "True" if risky_topk2 else "False"
    for position_bin, token_class, residual_norm_bin in (
        ("even", "common_target", "low"),
        ("odd", "rare_target", "mid"),
    ):
        variants = [
            ("promoted_contextual_topk2", topk2_logit_mse, topk2_churn),
            ("rank_matched_contextual_topk1", 0.1, "False"),
            ("random_fixed_topk2", 0.7, "False"),
        ]
        if include_dense:
            variants.append(("norm_matched_dense_active_rank", 0.2, ""))
        for variant, logit_mse, support_churn in variants:
            rows.append(
                {
                    "variant": variant,
                    "position_bin": position_bin,
                    "token_class": token_class,
                    "residual_norm_bin": residual_norm_bin,
                    "ce_abs_delta": logit_mse * 2,
                    "logit_mse": logit_mse,
                    "symmetric_kl": logit_mse / 2,
                    "residual_delta_l2": logit_mse * 10,
                    "support_churn": support_churn,
                }
            )
    _write_csv(path / "per_token_commutator.csv", list(rows[0]), rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
