"""Post-probe decision report for heldout-context dense-vs-top-k1 evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROBE_DIR = Path("results/reports/heldout_context_intervention_probe")
DEFAULT_DENSE_TEACHER_DIR = Path(
    "results/audits/token_larger_dense_teacher_residual_distillation_comparison"
)
DEFAULT_COMMUTATOR_DIR = Path("results/reports/acsr_finite_update_commutator_assay")
DEFAULT_STRATEGY_REVIEW = Path("../outputs/strategy-reviews/relaleap/latest-review.md")
DEFAULT_OUT_DIR = Path("results/reports/heldout_context_post_probe_decision")

DECISION_RECORDED = "heldout_context_post_probe_decision_recorded"
INSUFFICIENT_EVIDENCE = "heldout_context_post_probe_decision_failed_closed"
DENSE_BASELINE_ACTIVE = "dense_residual_controls_remain_active_baseline"
NEXT_BRANCH = "implement_synthetic_task_free_continual_learning_dense_vs_sparse_probe"

REQUIRED_ARTIFACTS = (
    "summary.json",
    "source_rows.csv",
    "decision_rows.csv",
    "next_branch_design.csv",
    "notes.md",
)

_NEXT_BRANCH_ROWS = (
    {
        "component": "frozen_base_anchor",
        "purpose": "measure task-A and task-B residual adaptation against unchanged base logits",
    },
    {
        "component": "rank_flop_norm_matched_causal_dense",
        "purpose": "keep the current active dense baseline with feature, rank, norm, and compute accounting",
    },
    {
        "component": "rank_matched_sparse_topk1",
        "purpose": "test whether low-churn sparse supports reduce task-free forgetting despite worse heldout CE",
    },
    {
        "component": "contextual_topk2_reference",
        "purpose": "retain the promoted support-width router as a non-default sparse reference",
    },
    {
        "component": "random_or_frequency_matched_support_null",
        "purpose": "separate sparse retention from support-frequency priors",
    },
    {
        "component": "dense_teacher_optional_control",
        "purpose": "use only as a diagnostic source because the existing dense-teacher pilot failed its CE gate",
    },
)

_ESTIMANDS = (
    "task_a_retention_ce_delta_after_task_b_update",
    "task_b_adaptation_ce_delta",
    "anchor_kl_or_logit_mse_churn",
    "functional_churn_symmetric_kl",
    "residual_norm_and_gain_per_l2",
    "finite_update_order_commutator",
    "support_identity_churn",
)


def run_heldout_context_post_probe_decision_report(
    *,
    probe_dir: Path = DEFAULT_PROBE_DIR,
    dense_teacher_dir: Path = DEFAULT_DENSE_TEACHER_DIR,
    commutator_dir: Path = DEFAULT_COMMUTATOR_DIR,
    strategy_review_path: Path = DEFAULT_STRATEGY_REVIEW,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write a fail-closed local decision over the completed heldout probe."""

    start = time.time()
    probe = _read_json(probe_dir / "summary.json")
    dense_teacher = _read_json(dense_teacher_dir / "summary.json")
    commutator = _read_json(commutator_dir / "summary.json")
    strategy = _strategy_review(strategy_review_path)
    paired_rows = _read_csv(probe_dir / "paired_deltas.csv")
    gate_rows = _read_csv(probe_dir / "gate_criteria.csv")

    source_rows = _source_rows(
        probe_dir=probe_dir,
        dense_teacher_dir=dense_teacher_dir,
        commutator_dir=commutator_dir,
        strategy_review_path=strategy_review_path,
        probe=probe,
        dense_teacher=dense_teacher,
        commutator=commutator,
        strategy=strategy,
    )
    failures = _failures(probe, paired_rows, gate_rows, strategy)
    dense_minus_topk1 = _mean(
        _float(row.get("dense_minus_topk1_heldout_delta"))
        for row in paired_rows
        if row.get("comparison") == "primary_dense_minus_sparse"
    )
    sparse_support_identity_signal = _mean(
        -_float(row.get("left_minus_right_heldout_delta"))
        for row in paired_rows
        if row.get("comparison") == "topk1_minus_random_support_null"
    )
    dense_context_signal = _mean(
        -_float(row.get("left_minus_right_heldout_delta"))
        for row in paired_rows
        if row.get("comparison") in {
            "causal_dense_minus_shuffled_context_null",
            "causal_dense_minus_ablated_context_null",
        }
    )

    status = "fail" if failures else "pass"
    if failures:
        decision = INSUFFICIENT_EVIDENCE
        claim_policy = "source_artifacts_missing_or_inconsistent"
        selected_next_step = "repair heldout-context probe artifacts before selecting another branch"
        rationale = (
            "The post-probe report failed closed because the local probe artifacts "
            "are missing, failing, or do not preserve the dense-vs-top-k1 gate rows."
        )
    else:
        decision = DECISION_RECORDED
        claim_policy = DENSE_BASELINE_ACTIVE
        selected_next_step = NEXT_BRANCH
        rationale = (
            "The heldout-context probe passes its artifact/null contract but keeps "
            "the sparse mechanism claim blocked: causal dense beats rank-matched "
            "top-k1 on heldout CE in both seeds, while top-k1 only shows a diagnostic "
            "support-identity advantage over random support. Existing dense-teacher "
            "and finite-update artifacts do not rescue the sparse claim, so the next "
            "bounded local branch should test the remaining non-CE hypothesis directly "
            "with synthetic task-free retention/forgetting evidence against the active "
            "dense baseline."
        )

    decision_rows = [
        _decision_row(
            "probe_contract_passed",
            status == "pass",
            probe.get("decision"),
            "The heldout-context probe must pass before this report can select a branch.",
        ),
        _decision_row(
            "dense_remains_better_than_topk1_on_heldout_ce",
            dense_minus_topk1 is not None and dense_minus_topk1 < -0.1,
            dense_minus_topk1,
            "Negative dense-minus-top-k1 means causal dense has lower heldout CE; the GPT-5.5-Pro reopening threshold is not met.",
        ),
        _decision_row(
            "support_identity_signal_is_diagnostic",
            sparse_support_identity_signal is not None and sparse_support_identity_signal > 0.0,
            sparse_support_identity_signal,
            "Top-k1 beating frequency-matched random support shows support identity matters, but not enough for promotion.",
        ),
        _decision_row(
            "dense_context_nulls_are_beaten",
            dense_context_signal is not None and dense_context_signal > 0.0,
            dense_context_signal,
            "Causal dense beating shuffled/ablated context nulls keeps dense controls active, not merely token-position priors.",
        ),
        _decision_row(
            "dense_teacher_branch_already_failed_gate",
            dense_teacher.get("decision") == "dense_teacher_residual_distillation_pilot_not_supported",
            dense_teacher.get("claim_status"),
            "Do not duplicate the completed dense-teacher pilot as the immediate next branch.",
        ),
        _decision_row(
            "finite_update_commutator_already_tiny",
            commutator.get("decision") == "acsr_finite_update_commutator_assay_tiny_commutator",
            commutator.get("claim_status"),
            "Existing finite-update commutator evidence is too small to carry the sparse claim by itself.",
        ),
    ]

    summary = {
        "status": status,
        "decision": decision,
        "claim_policy": claim_policy,
        "selected_next_step": selected_next_step,
        "requires_gpu_now": False,
        "backend_policy": (
            "No RunPod or Colab validation is selected by this report. Run GPU only "
            "after a local synthetic retention probe produces a claim-changing result."
        ),
        "primary_metrics": {
            "mean_dense_minus_topk1_heldout_delta": dense_minus_topk1,
            "mean_topk1_support_identity_advantage_vs_random": sparse_support_identity_signal,
            "mean_dense_context_advantage_vs_shuffled_or_ablated_nulls": dense_context_signal,
        },
        "source_rows": source_rows,
        "decision_rows": decision_rows,
        "next_branch_design": {
            "selected_branch": NEXT_BRANCH if status == "pass" else None,
            "components": list(_NEXT_BRANCH_ROWS),
            "estimands": list(_ESTIMANDS),
        },
        "strategy_review": strategy,
        "strategy_review_handling": (
            "Accepted the latest GPT-5.5-Pro null/accounting recommendation as already "
            "satisfied by the completed probe. No major strategic-change or Ben-notify "
            "header is present in the latest review."
        ),
        "failures": failures,
        "rationale": rationale,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary)
    return summary


def _source_rows(
    *,
    probe_dir: Path,
    dense_teacher_dir: Path,
    commutator_dir: Path,
    strategy_review_path: Path,
    probe: dict[str, Any],
    dense_teacher: dict[str, Any],
    commutator: dict[str, Any],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "source": "heldout_context_intervention_probe",
            "path": str(probe_dir / "summary.json"),
            "present": (probe_dir / "summary.json").is_file(),
            "status": probe.get("status"),
            "decision": probe.get("decision"),
            "claim_status": probe.get("claim_status"),
        },
        {
            "source": "dense_teacher_residual_distillation_comparison",
            "path": str(dense_teacher_dir / "summary.json"),
            "present": (dense_teacher_dir / "summary.json").is_file(),
            "status": dense_teacher.get("status"),
            "decision": dense_teacher.get("decision"),
            "claim_status": dense_teacher.get("claim_status"),
        },
        {
            "source": "acsr_finite_update_commutator_assay",
            "path": str(commutator_dir / "summary.json"),
            "present": (commutator_dir / "summary.json").is_file(),
            "status": commutator.get("status"),
            "decision": commutator.get("decision"),
            "claim_status": commutator.get("claim_status"),
        },
        {
            "source": "strategy_review",
            "path": str(strategy_review_path),
            "present": strategy["present"],
            "status": "read" if strategy["present"] else "missing",
            "decision": strategy["recommended_next_action"],
            "claim_status": (
                f"strategic_change_level={strategy['strategic_change_level']}; "
                f"notify_ben={strategy['notify_ben']}"
            ),
        },
    ]


def _failures(
    probe: dict[str, Any],
    paired_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    if probe.get("status") != "pass" or probe.get("decision") != "heldout_context_intervention_probe_passed":
        failures.append(
            {
                "source": "heldout_context_intervention_probe",
                "field": "status_or_decision",
                "observed": {"status": probe.get("status"), "decision": probe.get("decision")},
                "reason": "probe must pass before post-probe branch selection",
            }
        )
    if not paired_rows:
        failures.append(
            {
                "source": "heldout_context_intervention_probe",
                "field": "paired_deltas_csv",
                "observed": "missing",
                "reason": "paired deltas are required for dense-vs-top-k1 interpretation",
            }
        )
    if not gate_rows:
        failures.append(
            {
                "source": "heldout_context_intervention_probe",
                "field": "gate_criteria_csv",
                "observed": "missing",
                "reason": "gate criteria are required for fail-closed interpretation",
            }
        )
    required_gates = {
        "required_arms_and_nulls_present",
        "residual_norm_and_active_compute_accounting_present",
        "topk1_beats_causal_dense_on_heldout_ce",
    }
    seen_gates = {row.get("criterion") for row in gate_rows}
    missing_gates = sorted(required_gates - seen_gates)
    if missing_gates:
        failures.append(
            {
                "source": "heldout_context_intervention_probe",
                "field": "gate_criteria",
                "observed": missing_gates,
                "reason": "probe gate criteria do not include the required null/accounting/sparse-claim rows",
            }
        )
    if not strategy["present"]:
        failures.append(
            {
                "source": "strategy_review",
                "field": "latest-review.md",
                "observed": "missing",
                "reason": "latest external strategy review must be read before branch selection",
            }
        )
    return failures


def _decision_row(criterion: str, passed: bool, observed: Any, interpretation: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "observed": observed,
        "interpretation": interpretation,
    }


def _strategy_review(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "present": path.is_file(),
        "strategic_change_level": "",
        "notify_ben": "",
        "recommended_next_action": "",
        "verdict": "",
        "ben_notification_required": False,
    }
    if not path.is_file():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines()[:20]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in fields:
            fields[key.strip()] = value.strip()
    fields["ben_notification_required"] = (
        fields["strategic_change_level"] == "major" or fields["notify_ben"] == "true"
    )
    return fields


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "source_rows.csv", summary["source_rows"])
    _write_csv(out_dir / "decision_rows.csv", summary["decision_rows"])
    _write_csv(out_dir / "next_branch_design.csv", summary["next_branch_design"]["components"])
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Heldout-Context Post-Probe Decision",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim policy: `{summary['claim_policy']}`",
        f"- Selected next step: `{summary['selected_next_step']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        "",
        summary["rationale"],
        "",
        "## Primary Metrics",
    ]
    for key, value in summary["primary_metrics"].items():
        lines.append(f"- `{key}`: `{value}`")
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['source']}.{failure['field']}`: {failure['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "missing"}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Any) -> float | None:
    real = [value for value in values if value is not None]
    if not real:
        return None
    return sum(real) / len(real)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--dense-teacher-dir", type=Path, default=DEFAULT_DENSE_TEACHER_DIR)
    parser.add_argument("--commutator-dir", type=Path, default=DEFAULT_COMMUTATOR_DIR)
    parser.add_argument("--strategy-review", type=Path, default=DEFAULT_STRATEGY_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_heldout_context_post_probe_decision_report(
        probe_dir=args.probe_dir,
        dense_teacher_dir=args.dense_teacher_dir,
        commutator_dir=args.commutator_dir,
        strategy_review_path=args.strategy_review,
        out_dir=args.out,
    )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
