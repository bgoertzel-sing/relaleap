"""Backend provenance manifest for the active top-k-1 functional-retention packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from relaleap.experiments.active_topk1_functional_retention_audit import (
    FUNCTIONAL_RETENTION_BRACKET_ONLY,
)
from relaleap.experiments.active_topk1_singleton_reconciliation_audit import (
    CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
)


DEFAULT_LOCAL_FUNCTIONAL_RETENTION_DIR = Path(
    "results/reports/token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_RUNPOD_FUNCTIONAL_RETENTION_DIR = Path(
    "results/runpod_fetch/reports/local_checked_runpod_token_larger_active_topk1_functional_retention_audit"
)
DEFAULT_LOCAL_PROBE_DIRS = (
    Path("results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe"),
    Path(
        "results/audits/token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_RUNPOD_PROBE_DIRS = (
    Path(
        "results/runpod_fetch/audits/runpod_token_larger_active_rank_matched_topk1_retention_churn_probe"
    ),
    Path(
        "results/runpod_fetch/audits/runpod_token_larger_active_rank_matched_topk1_retention_churn_probe_seed2"
    ),
)
DEFAULT_LOCAL_SINGLETON_RECONCILIATION_DIR = Path(
    "results/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit"
)
DEFAULT_RUNPOD_SINGLETON_RECONCILIATION_DIR = Path(
    "results/runpod_fetch/audits/token_larger_active_rank_matched_topk1_singleton_reconciliation_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/reports/token_larger_active_topk1_backend_provenance_manifest"
)

ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED = (
    "active_topk1_backend_provenance_established"
)
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def run_active_topk1_backend_provenance_manifest(
    *,
    local_functional_retention_dir: Path = DEFAULT_LOCAL_FUNCTIONAL_RETENTION_DIR,
    runpod_functional_retention_dir: Path = DEFAULT_RUNPOD_FUNCTIONAL_RETENTION_DIR,
    local_probe_dirs: tuple[Path, ...] = DEFAULT_LOCAL_PROBE_DIRS,
    runpod_probe_dirs: tuple[Path, ...] = DEFAULT_RUNPOD_PROBE_DIRS,
    local_singleton_reconciliation_dir: Path = DEFAULT_LOCAL_SINGLETON_RECONCILIATION_DIR,
    runpod_singleton_reconciliation_dir: Path = DEFAULT_RUNPOD_SINGLETON_RECONCILIATION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a compact manifest tying local and RunPod bracket evidence together."""

    start = time.time()
    backend_rows = [
        _backend_row(
            backend="local",
            functional_retention_dir=local_functional_retention_dir,
            probe_dirs=local_probe_dirs,
            singleton_reconciliation_dir=local_singleton_reconciliation_dir,
        ),
        _backend_row(
            backend="runpod",
            functional_retention_dir=runpod_functional_retention_dir,
            probe_dirs=runpod_probe_dirs,
            singleton_reconciliation_dir=runpod_singleton_reconciliation_dir,
        ),
    ]
    artifact_rows = [
        artifact
        for backend in backend_rows
        for artifact in backend["artifacts"]
    ]
    failures = [
        failure
        for backend in backend_rows
        for failure in _backend_failures(backend)
    ]
    if not _same_nonempty(
        backend_rows, ("functional_retention_decision", "claim_status")
    ):
        failures.append(
            {
                "field": "backend_decision_match",
                "expected": "local and runpod functional-retention decisions match",
                "actual": {
                    row["backend"]: {
                        "decision": row["functional_retention_decision"],
                        "claim_status": row["claim_status"],
                    }
                    for row in backend_rows
                },
            }
        )

    if failures:
        status = "fail"
        decision = INSUFFICIENT_EVIDENCE
        rationale = (
            "The backend provenance manifest could not establish the active "
            "top-k-1 packet because one or more local/RunPod summaries, probe "
            "packets, singleton reconciliation packets, or decision fields are "
            "missing or inconsistent."
        )
        next_step = (
            "repair the missing provenance source packet or rerun the local "
            "artifact check before using the RunPod repeat as evidence"
        )
    else:
        status = "pass"
        decision = ACTIVE_TOPK1_BACKEND_PROVENANCE_ESTABLISHED
        rationale = (
            "Local and fetched RunPod functional-retention packets both pass as "
            "the same bracket-only active top-k-1 decision, with the same "
            "context-gated singleton-efficacy claim status. The manifest records "
            "the source probe and singleton packet hashes so the backend repeat "
            "can be audited without relying on notebook or terminal state."
        )
        next_step = (
            "use this manifest as the provenance anchor for the active top-k-1 "
            "functional-retention bracket; do not broaden it into a reusable "
            "singleton causal-retention claim while off-context interference remains"
        )

    summary = {
        "status": status,
        "decision": decision,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "backend_rows": [
            {key: value for key, value in row.items() if key != "artifacts"}
            for row in backend_rows
        ],
        "artifact_manifest": artifact_rows,
        "failures": failures,
        "rationale": rationale,
        "next_step": next_step,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "artifact_manifest_csv": str(out_dir / "artifact_manifest.csv"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_artifact_manifest(out_dir / "artifact_manifest.csv", artifact_rows)
    _write_notes(out_dir / "notes.md", summary)
    return summary


def _backend_row(
    *,
    backend: str,
    functional_retention_dir: Path,
    probe_dirs: tuple[Path, ...],
    singleton_reconciliation_dir: Path,
) -> dict[str, Any]:
    functional_summary = _read_json_object(functional_retention_dir / "summary.json")
    singleton_summary = _read_json_object(singleton_reconciliation_dir / "summary.json")
    artifacts = [
        _artifact_row(backend, "functional_retention_summary", functional_retention_dir / "summary.json"),
        _artifact_row(backend, "functional_retention_packet_metrics", functional_retention_dir / "packet_metrics.csv"),
        _artifact_row(backend, "functional_retention_notes", functional_retention_dir / "notes.md"),
        _artifact_row(backend, "singleton_reconciliation_summary", singleton_reconciliation_dir / "summary.json"),
        _artifact_row(
            backend,
            "singleton_reconciliation_by_context",
            singleton_reconciliation_dir / "singleton_reconciliation_by_context.csv",
        ),
        _artifact_row(
            backend,
            "singleton_reconciliation_by_stratum",
            singleton_reconciliation_dir / "singleton_reconciliation_by_stratum.csv",
        ),
    ]
    for index, probe_dir in enumerate(probe_dirs, start=1):
        artifacts.append(
            _artifact_row(
                backend,
                f"retention_churn_probe_seed{index}_summary",
                probe_dir / "summary.json",
            )
        )
    return {
        "backend": backend,
        "functional_retention_dir": str(functional_retention_dir),
        "functional_retention_status": functional_summary.get("status"),
        "functional_retention_decision": functional_summary.get("decision"),
        "claim_status": functional_summary.get("claim_status"),
        "singleton_reconciliation_dir": str(singleton_reconciliation_dir),
        "singleton_reconciliation_status": singleton_summary.get("status"),
        "singleton_reconciliation_decision": singleton_summary.get("decision"),
        "probe_dirs": [str(path) for path in probe_dirs],
        "probe_count": len(probe_dirs),
        "artifacts_present": all(row["present"] for row in artifacts),
        "artifacts": artifacts,
    }


def _backend_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    backend = row["backend"]
    expected_fields = {
        "functional_retention_status": "pass",
        "functional_retention_decision": FUNCTIONAL_RETENTION_BRACKET_ONLY,
        "claim_status": CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE,
        "singleton_reconciliation_status": "pass",
        "singleton_reconciliation_decision": (
            CONTEXT_GATED_SINGLETON_EFFICACY_WITH_OFFCONTEXT_INTERFERENCE
        ),
    }
    for field, expected in expected_fields.items():
        if row.get(field) != expected:
            failures.append(
                {
                    "backend": backend,
                    "field": field,
                    "expected": expected,
                    "actual": row.get(field),
                }
            )
    for artifact in row["artifacts"]:
        if not artifact["present"]:
            failures.append(
                {
                    "backend": backend,
                    "field": artifact["role"],
                    "expected": "file exists",
                    "actual": "missing",
                    "path": artifact["path"],
                }
            )
    return failures


def _artifact_row(backend: str, role: str, path: Path) -> dict[str, Any]:
    return {
        "backend": backend,
        "role": role,
        "path": str(path),
        "present": path.is_file(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _same_nonempty(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    for field in fields:
        values = [row.get(field) for row in rows]
        if any(value in (None, "") for value in values) or len(set(values)) != 1:
            return False
    return True


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _write_artifact_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["backend", "role", "path", "present", "sha256", "size_bytes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Active Top-k-1 Backend Provenance Manifest",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Backends",
    ]
    for row in summary["backend_rows"]:
        lines.extend(
            [
                "",
                f"- Backend: `{row['backend']}`",
                f"  - Functional-retention decision: `{row['functional_retention_decision']}`",
                f"  - Claim status: `{row['claim_status']}`",
                f"  - Singleton reconciliation: `{row['singleton_reconciliation_decision']}`",
                f"  - Probe count: `{row['probe_count']}`",
                f"  - Artifacts present: `{row['artifacts_present']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Rationale",
            "",
            summary["rationale"],
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-functional-retention-dir",
        type=Path,
        default=DEFAULT_LOCAL_FUNCTIONAL_RETENTION_DIR,
    )
    parser.add_argument(
        "--runpod-functional-retention-dir",
        type=Path,
        default=DEFAULT_RUNPOD_FUNCTIONAL_RETENTION_DIR,
    )
    parser.add_argument(
        "--local-probe-dir",
        type=Path,
        action="append",
        dest="local_probe_dirs",
    )
    parser.add_argument(
        "--runpod-probe-dir",
        type=Path,
        action="append",
        dest="runpod_probe_dirs",
    )
    parser.add_argument(
        "--local-singleton-reconciliation-dir",
        type=Path,
        default=DEFAULT_LOCAL_SINGLETON_RECONCILIATION_DIR,
    )
    parser.add_argument(
        "--runpod-singleton-reconciliation-dir",
        type=Path,
        default=DEFAULT_RUNPOD_SINGLETON_RECONCILIATION_DIR,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_active_topk1_backend_provenance_manifest(
        local_functional_retention_dir=args.local_functional_retention_dir,
        runpod_functional_retention_dir=args.runpod_functional_retention_dir,
        local_probe_dirs=tuple(args.local_probe_dirs)
        if args.local_probe_dirs
        else DEFAULT_LOCAL_PROBE_DIRS,
        runpod_probe_dirs=tuple(args.runpod_probe_dirs)
        if args.runpod_probe_dirs
        else DEFAULT_RUNPOD_PROBE_DIRS,
        local_singleton_reconciliation_dir=args.local_singleton_reconciliation_dir,
        runpod_singleton_reconciliation_dir=args.runpod_singleton_reconciliation_dir,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "decision": summary["decision"],
                "backend_rows": summary["backend_rows"],
                "failures": summary["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
