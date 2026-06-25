from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaleap.experiments.active_topk1_backend_provenance_manifest import (
    ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED,
    INSUFFICIENT_EVIDENCE,
    run_active_topk1_backend_provenance_manifest,
)
from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


class ActiveTopk1BackendProvenanceManifestTest(unittest.TestCase):
    def test_manifest_records_matching_local_and_runpod_packet_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = _write_backend(root / "local")
            runpod = _write_backend(root / "runpod")

            summary = run_active_topk1_backend_provenance_manifest(
                local_functional_retention_dir=local["functional"],
                runpod_functional_retention_dir=runpod["functional"],
                local_probe_dirs=local["probes"],
                runpod_probe_dirs=runpod["probes"],
                local_singleton_reconciliation_dir=local["singleton"],
                runpod_singleton_reconciliation_dir=runpod["singleton"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["decision"], ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED)
            self.assertEqual(summary["failures"], [])
            self.assertEqual(len(summary["backend_rows"]), 2)
            self.assertTrue(all(row["artifacts_present"] for row in summary["backend_rows"]))
            self.assertTrue(all(row["sha256"] for row in summary["artifact_manifest"]))
            self.assertTrue((root / "report" / "summary.json").is_file())
            self.assertTrue((root / "report" / "artifact_manifest.csv").is_file())
            self.assertTrue((root / "report" / "notes.md").is_file())

    def test_manifest_fails_closed_when_runpod_probe_summary_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local = _write_backend(root / "local")
            runpod = _write_backend(root / "runpod")
            (runpod["probes"][1] / "summary.json").unlink()

            summary = run_active_topk1_backend_provenance_manifest(
                local_functional_retention_dir=local["functional"],
                runpod_functional_retention_dir=runpod["functional"],
                local_probe_dirs=local["probes"],
                runpod_probe_dirs=runpod["probes"],
                local_singleton_reconciliation_dir=local["singleton"],
                runpod_singleton_reconciliation_dir=runpod["singleton"],
                out_dir=root / "report",
            )

            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["decision"], INSUFFICIENT_EVIDENCE)
            fields = {failure["field"] for failure in summary["failures"]}
            self.assertIn("retention_churn_probe_seed2_summary", fields)


def _write_backend(root: Path) -> dict[str, object]:
    functional = root / "functional"
    singleton = root / "singleton"
    probes = (root / "probe_seed1", root / "probe_seed2")
    functional.mkdir(parents=True)
    singleton.mkdir(parents=True)
    for probe in probes:
        probe.mkdir(parents=True)
        _write_json(
            probe / "summary.json",
            {
                "status": "pass",
                "decision": "active_topk1_retention_churn_probe_established",
            },
        )
    _write_json(
        functional / "summary.json",
        {
            "status": "pass",
            "decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
            "claim_status": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        },
    )
    (functional / "packet_metrics.csv").write_text("packet,status\nseed1,pass\n", encoding="utf-8")
    (functional / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _write_json(
        singleton / "summary.json",
        {
            "status": "pass",
            "decision": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        },
    )
    (singleton / "singleton_reconciliation_by_context.csv").write_text(
        "context,status\n1,pass\n",
        encoding="utf-8",
    )
    (singleton / "singleton_reconciliation_by_stratum.csv").write_text(
        "stratum,status\n1,pass\n",
        encoding="utf-8",
    )
    return {"functional": functional, "singleton": singleton, "probes": probes}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
