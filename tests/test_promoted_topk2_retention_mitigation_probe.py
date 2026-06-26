from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from relaleap.experiments.promoted_topk2_retention_mitigation_probe import (
    INSUFFICIENT_EVIDENCE,
    RETENTION_MITIGATION_CANDIDATE_FOUND,
    RETENTION_MITIGATION_NOT_ESTABLISHED,
    run_promoted_topk2_retention_mitigation_probe,
)


class PromotedTopk2RetentionMitigationProbeTest(unittest.TestCase):
    def test_runs_extended_microtest_and_writes_gate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
run:
  experiment_id: test_retention_mitigation_probe
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

            summary = run_promoted_topk2_retention_mitigation_probe(
                config_path=config_path,
                out_dir=root / "probe",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertIn(
                summary["decision"],
                {
                    RETENTION_MITIGATION_CANDIDATE_FOUND,
                    RETENTION_MITIGATION_NOT_ESTABLISHED,
                },
            )
            self.assertEqual(summary["metrics"]["control_count"], 3)
            self.assertEqual(summary["metrics"]["mitigation_count"], 2)
            self.assertEqual(len(summary["mitigation_rows"]), 2)
            self.assertTrue((root / "probe" / "summary.json").is_file())
            self.assertTrue((root / "probe" / "mitigation_rows.csv").is_file())
            self.assertTrue((root / "probe" / "variant_metrics.csv").is_file())
            saved = json.loads((root / "probe" / "summary.json").read_text())
            self.assertEqual(saved["decision"], summary["decision"])

    def test_finds_candidate_when_mocked_mitigation_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "relaleap.experiments.promoted_topk2_retention_mitigation_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.side_effect = lambda config_path, out_dir, **kwargs: _mock_microtest(
                    out_dir
                )

                summary = run_promoted_topk2_retention_mitigation_probe(
                    config_path=root / "config.yaml",
                    out_dir=root / "probe",
                )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], RETENTION_MITIGATION_CANDIDATE_FOUND)
            self.assertEqual(
                summary["mitigation_rows"][0][
                    "commutator_anchor_logit_mse_reduction_fraction"
                ],
                0.75,
            )

    def test_fails_closed_when_rows_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch(
                "relaleap.experiments.promoted_topk2_retention_mitigation_probe.run_retention_churn_microtest"
            ) as mocked:
                mocked.return_value = {"status": "ok", "audit": {"variants": []}}

                summary = run_promoted_topk2_retention_mitigation_probe(
                    config_path=root / "config.yaml",
                    out_dir=root / "probe",
                )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)


def _mock_microtest(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variant_metrics.csv").write_text("", encoding="utf-8")
    variants = [
        _variant("promoted_contextual_topk2", commutator=0.24, transfer=1.0, used=10),
        _variant("rank_matched_contextual_topk1", commutator=0.01, transfer=1.1, used=10),
        _variant("random_fixed_topk2", commutator=0.30, transfer=0.7, used=10),
        _variant("norm_matched_dense_active_rank", commutator=0.08, transfer=0.5, used=""),
        _variant(
            "router_frozen_transfer_topk2",
            commutator=0.06,
            transfer=0.9,
            used=9,
            freeze=True,
        ),
        _variant(
            "gradient_clipped_contextual_topk2",
            commutator=0.22,
            transfer=0.9,
            used=10,
            clip=0.25,
        ),
    ]
    return {"status": "ok", "audit": {"variants": variants}}


def _variant(
    name: str,
    *,
    commutator: float,
    transfer: float,
    used: int | str,
    freeze: bool = False,
    clip: float | str = "",
) -> dict[str, object]:
    return {
        "variant": name,
        "commutator_anchor_logit_mse": commutator,
        "commutator_transfer_logit_mse": commutator,
        "transfer_ce_improvement": transfer,
        "anchor_used_columns_after_transfer": used,
        "anchor_support_churn_after_transfer": 0.5,
        "commutator_anchor_support_churn": 0.5,
        "freeze_router_during_transfer": freeze,
        "gradient_clip_norm": clip,
    }


if __name__ == "__main__":
    unittest.main()
