"""Design a fail-closed functional intervention for the causal-router pivot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PIVOT_AUDIT = Path("results/reports/acsr_causal_support_router_pivot_audit/summary.json")
DEFAULT_CAPACITY_AUDIT = Path("results/audits/acsr_causal_router_capacity_audit_local/summary.json")
DEFAULT_SAME_STUDENT_REPORT = Path(
    "results/reports/token_larger_causal_contextual_router_same_student_intervention_matrix/summary.json"
)
DEFAULT_SAME_STUDENT_MATRIX = Path(
    "results/reports/token_larger_causal_contextual_router_same_student_intervention_matrix/"
    "same_student_matrix.csv"
)
DEFAULT_ROUTER_VALUE_AUDIT = Path(
    "results/audits/token_larger_router_value_disentanglement_audit/summary.json"
)
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path(
    "results/reports/acsr_causal_support_router_functional_intervention_design"
)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_reports.csv",
    "intervention_design.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_acsr_causal_support_router_functional_intervention_design(
    *,
    pivot_audit: Path = DEFAULT_PIVOT_AUDIT,
    capacity_audit: Path = DEFAULT_CAPACITY_AUDIT,
    same_student_report: Path = DEFAULT_SAME_STUDENT_REPORT,
    same_student_matrix: Path = DEFAULT_SAME_STUDENT_MATRIX,
    router_value_audit: Path = DEFAULT_ROUTER_VALUE_AUDIT,
    strategy_review: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write the next local intervention design without promoting the mechanism."""

    start = time.time()
    sources = {
        "pivot_audit": _read_json_object(pivot_audit),
        "capacity_audit": _read_json_object(capacity_audit),
        "same_student_report": _read_json_object(same_student_report),
        "router_value_audit": _read_json_object(router_value_audit),
    }
    paths = {
        "pivot_audit": pivot_audit,
        "capacity_audit": capacity_audit,
        "same_student_report": same_student_report,
        "router_value_audit": router_value_audit,
    }
    same_student_rows = _read_csv_rows(same_student_matrix)
    review = _strategy_review(strategy_review)
    source_rows = _source_rows(sources, paths, same_student_matrix, same_student_rows, review)
    design_rows = _design_rows(sources, same_student_rows)
    gate_rows = _gate_rows(sources, same_student_rows, review, design_rows)
    failures = [row for row in gate_rows if not row["passed"]]
    status = "pass" if not failures else "fail"
    summary = {
        "status": status,
        "decision": (
            "causal_support_router_functional_intervention_design_recorded"
            if status == "pass"
            else "causal_support_router_functional_intervention_design_failed_closed"
        ),
        "claim_status": "design_only_mechanism_not_established",
        "selected_next_step": _selected_next_step(sources),
        "next_command": (
            "./.venv-conda/bin/python -m relaleap.experiments.acsr_causal_router_capacity_audit"
        ),
        "source_reports": source_rows,
        "intervention_design": design_rows,
        "gate_criteria": gate_rows,
        "failures": failures,
        "strategy_review": review,
        "direction_shift": _direction_shift(review),
        "claim_boundaries": {
            "supported": [
                "ACSR-as-anticipation remains blocked by the parameter-matched causal MLP control",
                "same-student token-position-null support forcing exists and blocks the functional claim",
                "router/value disentanglement evidence shows learned values and support choices are entangled",
                "the next intervention can be specified from existing artifact schemas",
            ],
            "not_supported": [
                "direct causal support-router mechanism",
                "ACSR promotion",
                "GPU repeat as a useful next step",
                "dual-student value/support mechanism claim, until cross-forcing beats nulls robustly",
            ],
        },
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    sources: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    same_student_matrix: Path,
    same_student_rows: list[dict[str, str]],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for name, source in sources.items():
        rows.append(
            {
                "source": name,
                "path": str(paths[name]),
                "present": bool(source),
                "status": source.get("status", "missing"),
                "decision": source.get("decision", ""),
                "claim_status": source.get("claim_status", ""),
                "git_commit": source.get("git_commit", ""),
            }
        )
    rows.append(
        {
            "source": "same_student_matrix",
            "path": str(same_student_matrix),
            "present": same_student_matrix.is_file(),
            "status": "present" if same_student_rows else "missing",
            "decision": f"rows={len(same_student_rows)}",
            "claim_status": "same_student_support_forcing_arms",
            "git_commit": "",
        }
    )
    rows.append(
        {
            "source": "strategy_review",
            "path": review["path"],
            "present": review["present"],
            "status": "read" if review["present"] else "missing",
            "decision": review["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={review['strategic_change_level']}; "
                f"notify_ben={review['notify_ben']}"
            ),
            "git_commit": "",
        }
    )
    return rows


def _design_rows(
    sources: dict[str, dict[str, Any]],
    same_student_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    same_student = sources["same_student_report"].get("key_metrics", {})
    capacity = sources["capacity_audit"].get("aggregate_metrics", {})
    router_value = sources["router_value_audit"].get("evidence", {})
    all_tokens = _matrix_row(same_student_rows, "all_tokens")
    disagreement = _matrix_row(same_student_rows, "teacher_student_disagreement_tokens")
    return [
        {
            "intervention": "same_student_token_position_null_forcing",
            "source_basis": "same_student_intervention_matrix",
            "available_now": True,
            "primary_metric": "teacher_minus_token_position_null_gain_all_tokens",
            "current_value": same_student.get("teacher_minus_token_position_null_gain_all_tokens", ""),
            "required_for_mechanism": "teacher support must beat token-position null through identical student values",
            "current_interpretation": "blocked; teacher support is not functionally better enough and teacher forcing worsens all-token loss",
        },
        {
            "intervention": "dual_student_cross_forcing",
            "source_basis": "ACSR source-packet dual_student_cross_forcing.csv",
            "available_now": bool(capacity.get("dual_student_cross_forcing_available")),
            "primary_metric": "mean_partner_support_delta_vs_token_position_null",
            "current_value": capacity.get("mean_partner_support_delta_vs_token_position_null", ""),
            "required_for_mechanism": (
                "causal-router supports must improve or preserve loss through independently "
                "trained values and ACSR supports must not explain the effect by value co-adaptation"
            ),
            "current_interpretation": (
                "available; interpret partner-transfer rows against token-position, shuffled, "
                "random, oracle, and teacher diagnostics before any promotion claim"
            ),
        },
        {
            "intervention": "oracle_regret_margin_stratification",
            "source_basis": "ACSR capacity audit plus same-student matrix",
            "available_now": bool(all_tokens and disagreement),
            "primary_metric": "all_token_and_disagreement_token_oracle_gap",
            "current_value": (
                f"all_tokens_oracle_gain={_value(all_tokens, 'gain_vs_student_router')}; "
                f"disagreement_oracle_gain={_value(disagreement, 'gain_vs_student_router')}"
            ),
            "required_for_mechanism": "support improvements should concentrate where oracle support has usable headroom",
            "current_interpretation": "oracle headroom exists, but teacher/token-position supports do not exploit it",
        },
        {
            "intervention": "capacity_matched_acsr_vs_direct_causal_router",
            "source_basis": "ACSR causal-router capacity audit",
            "available_now": True,
            "primary_metric": "mean_acsr_minus_parameter_matched_ce_loss",
            "current_value": capacity.get("mean_acsr_minus_parameter_matched_ce_loss", ""),
            "required_for_mechanism": "ACSR must beat the direct causal MLP control before anticipation claims reopen",
            "current_interpretation": "blocked; positive means ACSR is worse than the direct causal control",
        },
        {
            "intervention": "router_value_entanglement_guardrail",
            "source_basis": "router_value_disentanglement_audit",
            "available_now": True,
            "primary_metric": "value_only_fraction_of_full_vs_router_only_fraction_of_full",
            "current_value": (
                f"value_only={router_value.get('value_only_fraction_of_full', '')}; "
                f"router_only={router_value.get('router_only_fraction_of_full', '')}"
            ),
            "required_for_mechanism": "support-policy evidence must be reported separately from residual-value path effects",
            "current_interpretation": "value path dominates, so dual-student forcing is mandatory before mechanism promotion",
        },
    ]


def _gate_rows(
    sources: dict[str, dict[str, Any]],
    same_student_rows: list[dict[str, str]],
    review: dict[str, Any],
    design_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    capacity_status = sources["capacity_audit"].get("claim_status", "")
    pivot_status = sources["pivot_audit"].get("claim_status", "")
    same_student_status = sources["same_student_report"].get("claim_status", "")
    router_value_statuses = sources["router_value_audit"].get("claim_statuses", {})
    return [
        _criterion(
            "strategy_review_consumed",
            review["present"],
            "latest strategy review is present and read",
            {
                "strategic_change_level": review["strategic_change_level"],
                "notify_ben": review["notify_ben"],
            },
            "latest GPT-5.5-Pro strategy review was not consumed",
        ),
        _criterion(
            "acsr_anticipation_stays_blocked",
            capacity_status == "acsr_as_anticipation_blocked_by_capacity_matched_causal_router",
            "capacity audit must block ACSR-as-anticipation",
            capacity_status,
            "ACSR anticipation claim is not blocked by the capacity-matched control",
        ),
        _criterion(
            "pivot_audit_stays_fail_closed",
            pivot_status == "direct_causal_support_router_mechanism_not_established",
            "pivot audit must keep the causal-router mechanism unestablished",
            pivot_status,
            "pivot audit no longer marks the mechanism unestablished",
        ),
        _criterion(
            "same_student_token_position_null_available",
            bool(same_student_rows)
            and "same_student_token_position_null" in same_student_status,
            "same-student token-position-null report and matrix are available",
            {"claim_status": same_student_status, "rows": len(same_student_rows)},
            "same-student token-position-null intervention evidence is missing",
        ),
        _criterion(
            "router_value_entanglement_accounted",
            router_value_statuses.get("router_value_disentanglement")
            == "recorded_value_path_and_support_selection_entangled",
            "router/value disentanglement audit records value/support entanglement",
            router_value_statuses.get("router_value_disentanglement"),
            "router/value entanglement guardrail is missing",
        ),
        _criterion(
            "dual_student_design_explicit",
            any(row["intervention"] == "dual_student_cross_forcing" for row in design_rows),
            "design includes dual-student cross-forcing as the required next source extension",
            [row["intervention"] for row in design_rows],
            "dual-student cross-forcing is not in the design",
        ),
    ]


def _criterion(
    criterion: str,
    passed: bool,
    threshold: str,
    actual: Any,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": passed,
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _selected_next_step(sources: dict[str, dict[str, Any]]) -> str:
    capacity = sources["capacity_audit"].get("aggregate_metrics", {})
    if not capacity.get("dual_student_cross_forcing_available"):
        return "implement dual-student support cross-forcing in the causal-router source packet"
    return (
        "interpret dual-student cross-forcing transfer against token-position, "
        "shuffled, random, oracle, and teacher controls before any causal-router "
        "mechanism claim"
    )


def _matrix_row(rows: list[dict[str, str]], token_subset: str) -> dict[str, str]:
    for row in rows:
        if (
            row.get("token_subset") == token_subset
            and row.get("intervention") == "oracle_best_support_for_student"
        ):
            return row
    return {}


def _value(row: dict[str, str], key: str) -> str:
    return row.get(key, "") if row else ""


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strategy_review(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "present": False,
            "strategic_change_level": "",
            "notify_ben": "",
            "recommended_next_action": "",
            "verdict": "",
        }
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return {
        "path": str(path),
        "present": True,
        "strategic_change_level": header.get("strategic_change_level", ""),
        "notify_ben": header.get("notify_ben", ""),
        "recommended_next_action": header.get("recommended_next_action", ""),
        "verdict": header.get("verdict", ""),
    }


def _direction_shift(review: dict[str, Any]) -> str:
    if not review["present"]:
        return "No strategy review was available; fail closed."
    return (
        f"Accepted GPT-5.5-Pro direction: {review['verdict']} / "
        f"{review['recommended_next_action']} Ben should be notified: "
        f"{str(review['notify_ben']).lower()}."
    )


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_reports.csv", summary["source_reports"])
    _write_csv(out_dir / "intervention_design.csv", summary["intervention_design"])
    _write_csv(out_dir / "gate_criteria.csv", summary["gate_criteria"])
    _write_notes(out_dir / "notes.md", summary)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACSR Causal Support-Router Functional Intervention Design",
        "",
        f"Decision: `{summary['decision']}`.",
        f"Claim status: `{summary['claim_status']}`.",
        "",
        summary["direction_shift"],
        "",
        "## Design",
    ]
    for row in summary["intervention_design"]:
        lines.append(
            "- "
            f"{row['intervention']}: {row['current_interpretation']} "
            f"(metric `{row['primary_metric']}` = {row['current_value']})."
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            summary["selected_next_step"],
            "",
            "No GPU repeat is selected by this design report.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pivot-audit", type=Path, default=DEFAULT_PIVOT_AUDIT)
    parser.add_argument("--capacity-audit", type=Path, default=DEFAULT_CAPACITY_AUDIT)
    parser.add_argument("--same-student-report", type=Path, default=DEFAULT_SAME_STUDENT_REPORT)
    parser.add_argument("--same-student-matrix", type=Path, default=DEFAULT_SAME_STUDENT_MATRIX)
    parser.add_argument("--router-value-audit", type=Path, default=DEFAULT_ROUTER_VALUE_AUDIT)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_acsr_causal_support_router_functional_intervention_design(
        pivot_audit=args.pivot_audit,
        capacity_audit=args.capacity_audit,
        same_student_report=args.same_student_report,
        same_student_matrix=args.same_student_matrix,
        router_value_audit=args.router_value_audit,
        strategy_review=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
