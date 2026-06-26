from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.promoted_topk2_hub_value_composition_mitigation_probe import (
    HUB_VALUE_COMPOSITION_CANDIDATE_FOUND,
    HUB_VALUE_COMPOSITION_NOT_ESTABLISHED,
    INSUFFICIENT_EVIDENCE,
    run_promoted_topk2_hub_value_composition_mitigation_probe,
)


class PromotedTopk2HubValueCompositionMitigationProbeTest(unittest.TestCase):
    def test_runs_hub_microtest_and_writes_gate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = _write_config(root)
            localization = _write_localization(root)

            summary = run_promoted_topk2_hub_value_composition_mitigation_probe(
                config_path=config_path,
                out_dir=root / "gate",
                localization_report_path=localization,
                strategy_review_path=root / "missing-review.md",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {
                    HUB_VALUE_COMPOSITION_CANDIDATE_FOUND,
                    HUB_VALUE_COMPOSITION_NOT_ESTABLISHED,
                },
            )
            self.assertEqual(summary["metrics"]["control_count"], 3)
            self.assertEqual(summary["metrics"]["hub_value_composition_count"], 2)
            self.assertEqual(len(summary["hub_value_composition_rows"]), 2)
            self.assertEqual(
                summary["claim_statuses"]["topk2_causal_cooperation"],
                "not_supported",
            )
            self.assertTrue((root / "gate" / "summary.json").is_file())
            self.assertTrue(
                (root / "gate" / "hub_value_composition_rows.csv").is_file()
            )
            self.assertTrue((root / "gate" / "variant_metrics.csv").is_file())

    def test_finds_candidate_when_mocked_hub_variant_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            localization = _write_localization(root)
            with patch(
                "relaleap.experiments.promoted_topk2_hub_value_composition_mitigation_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(
                    out_dir
                )

                summary = run_promoted_topk2_hub_value_composition_mitigation_probe(
                    config_path=root / "config.yaml",
                    out_dir=root / "gate",
                    localization_report_path=localization,
                    strategy_review_path=root / "missing-review.md",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], HUB_VALUE_COMPOSITION_CANDIDATE_FOUND)
            self.assertEqual(
                summary["hub_value_composition_rows"][0][
                    "commutator_anchor_logit_mse_reduction_fraction"
                ],
                0.75,
            )
            self.assertEqual(
                summary["selected_next_action"],
                "runpod_hub_value_composition_validation",
            )
            saved = json.loads((root / "gate" / "summary.json").read_text())
            self.assertEqual(saved["decision"], summary["decision"])

    def test_fails_closed_when_localization_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "relaleap.experiments.promoted_topk2_hub_value_composition_mitigation_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(
                    out_dir
                )

                summary = run_promoted_topk2_hub_value_composition_mitigation_probe(
                    config_path=root / "config.yaml",
                    out_dir=root / "gate",
                    localization_report_path=root / "missing.json",
                    strategy_review_path=root / "missing-review.md",
                )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {
                (failure.get("source"), failure.get("field"))
                for failure in summary["failures"]
            }
            self.assertIn(
                ("pairwise_value_interaction_localization", "status"),
                fields,
            )


def _write_config(root: Path) -> Path:
    config_path = root / "config.yaml"
    config_path.write_text(
        """
run:
  experiment_id: test_hub_value_composition_probe
  seed: 1
  max_steps: 2

data:
  dataset: tiny_shakespeare_char
  seq_len: 16

training:
  residual_objective: supervised_ce

model:
  base:
    layers: 1
    hidden_dim: 32
  columns:
    num_columns: 4
    atoms_per_column: 2
    top_k: 2
    insertion_sites: 1
    support_router: contextual_mlp
    contextual_router_hidden_dim: 16
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_localization(root: Path) -> Path:
    path = root / "localization.json"
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "decision": "pairwise_value_interaction_localized_hub_family",
                "metrics": {
                    "dominant_column": 2,
                    "dominant_column_abs_synergy_share": 0.88,
                    "top3_pair_abs_synergy_share": 0.40,
                    "value_only_fraction_of_full": 1.25,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _mock_microtest(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant_metrics.csv").write_text("", encoding="utf-8")
    variants = [
        _variant("promoted_contextual_topk2", commutator=0.24, transfer=1.0, used=10),
        _variant("rank_matched_contextual_topk1", commutator=0.01, transfer=1.1, used=10),
        _variant("random_fixed_topk2", commutator=0.30, transfer=0.7, used=10),
        _variant("norm_matched_dense_active_rank", commutator=0.08, transfer=0.5, used=""),
        _variant(
            "hub_value_composition_w010_contextual_topk2",
            commutator=0.06,
            transfer=0.9,
            used=9,
            hub_weight=0.1,
            residual_l2=1.9,
        ),
        _variant(
            "hub_value_composition_w100_contextual_topk2",
            commutator=0.22,
            transfer=0.9,
            used=10,
            hub_weight=1.0,
            residual_l2=2.0,
        ),
    ]
    return {"status": "ok", "audit": {"variants": variants}}


def _variant(
    name: str,
    *,
    commutator: float,
    transfer: float,
    used: int | str,
    hub_weight: float | str = "",
    residual_l2: float = 2.0,
) -> dict[str, object]:
    return {
        "variant": name,
        "commutator_anchor_logit_mse": commutator,
        "commutator_transfer_logit_mse": commutator,
        "transfer_ce_improvement": transfer,
        "anchor_used_columns_after_transfer": used,
        "anchor_ce_drift": 0.0,
        "anchor_support_churn_after_transfer": 0.5,
        "commutator_anchor_support_churn": 0.5,
        "commutator_anchor_residual_stream_l2": residual_l2,
        "hub_value_composition_penalty_weight": hub_weight,
        "hub_value_composition_column": 2,
    }


if __name__ == "__main__":
    unittest.main()
