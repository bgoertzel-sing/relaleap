from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.anticipatory_contextual_support_routing_causal_mechanism_gate import (
    ACSR_CAUSAL_MECHANISM_SUPPORTED_NOT_PROMOTED,
    INSUFFICIENT_EVIDENCE,
    run_acsr_causal_mechanism_gate,
)


class AnticipatoryContextualSupportRoutingCausalMechanismGateTest(unittest.TestCase):
    def test_records_heldout_oracle_regret_functional_churn_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            packets = [root / f"packet{i}" for i in range(2)]
            for packet in packets:
                _write_packet(packet)
            previous_probe = root / "previous_probe.json"
            _write_previous_probe(previous_probe, passed=True)
            review = root / "latest-review.md"
            review.write_text(
                "\n".join(
                    [
                        "strategic_change_level: minor",
                        "notify_ben: false",
                        "recommended_next_action: Implement and run the local CPU ACSR smoke pilot with explicit leakage, shuffled-feature, token/position-only, same-student, and retention/churn gates before any GPU/backend replication.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = run_acsr_causal_mechanism_gate(
                audit_dirs=tuple(packets),
                previous_probe_path=previous_probe,
                strategy_review_path=review,
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(
                summary["decision"],
                ACSR_CAUSAL_MECHANISM_SUPPORTED_NOT_PROMOTED,
            )
            self.assertTrue(
                summary["gate_status"][
                    "passes_heldout_oracle_regret_functional_churn_gate"
                ]
            )
            self.assertEqual(len(summary["packet_rows"]), 2)
            self.assertTrue(summary["aggregate_rows"][0]["all_pass"])
            self.assertEqual(
                summary["deferred_or_rejected_recommendations"][0]["status"],
                "accepted_already_satisfied_and_extended",
            )
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "packet_gate_metrics.csv").is_file())
            self.assertTrue((root / "report" / "aggregate_gate_metrics.csv").is_file())
            self.assertTrue((root / "report" / "gate_criteria.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_fails_closed_when_token_position_has_lower_oracle_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            packet = root / "packet"
            _write_packet(packet, token_position_oracle_regret=0.01)
            previous_probe = root / "previous_probe.json"
            _write_previous_probe(previous_probe, passed=True)

            summary = run_acsr_causal_mechanism_gate(
                audit_dirs=(packet,),
                previous_probe_path=previous_probe,
                strategy_review_path=root / "missing-review.md",
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            failures = {failure.get("field") for failure in summary["failures"]}
            self.assertIn(
                "acsr_oracle_regret_not_worse_than_causal_and_nulls",
                failures,
            )


def _write_previous_probe(path: Path, *, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "decision": "previous_probe",
                "gate_status": {
                    "passes_cross_context_retention_churn_gate": passed,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_packet(
    path: Path,
    *,
    token_position_oracle_regret: float = 0.20,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "anticipatory_contextual_support_routing_smoke_completed",
                "gates": {"future_perturbation_invariance": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "router_metrics.csv").write_text(
        "\n".join(
            [
                "variant,oracle_regret",
                "acsr_mlp_predicted_future,0.02",
                "causal_feature_safe_contextual_topk2,0.18",
                f"token_position_only_predicted_features,{token_position_oracle_regret}",
                "shuffled_predicted_features,0.70",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "retention_churn_metrics.csv").write_text(
        "\n".join(
            [
                "phase,variant,anchor_logit_mse_after_transfer,teacher_logit_mse",
                "second_context_transfer,acsr_mlp_predicted_future,0.01,",
                "second_context_transfer,token_position_only_predicted_features,0.03,",
                "second_context_transfer,shuffled_predicted_features,0.19,",
                "fixed_context_teacher_reference,acsr_mlp_predicted_future,,0.001",
                "fixed_context_teacher_reference,token_position_only_predicted_features,,0.03",
                "fixed_context_teacher_reference,shuffled_predicted_features,,0.13",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "feature_perturbation.csv").write_text(
        "check,passed\nfuture_positions_do_not_change_prefix,true\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
