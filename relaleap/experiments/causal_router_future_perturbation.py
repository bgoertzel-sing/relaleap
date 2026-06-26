"""Future-perturbation evidence for the causal contextual support router."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_OUT_DIR = Path("results/reports/causal_router_future_perturbation")


def run_causal_router_future_perturbation(
    *, out_dir: Path = DEFAULT_OUT_DIR
) -> dict[str, Any]:
    """Write a fail-closed artifact for causal-router future invariance."""

    start = time.time()
    try:
        import torch

        from relaleap.smoke import ResidualColumns
    except Exception as exc:
        summary = _failure_summary(
            out_dir=out_dir,
            start=start,
            reason=f"torch_or_residual_import_failed: {exc}",
        )
        _write_artifacts(out_dir, summary)
        return summary

    torch.manual_seed(7)
    deterministic = _deterministic_contrast(torch, ResidualColumns)
    random_weight = _random_weight_check(torch, ResidualColumns)
    passed = (
        deterministic["causal_scores_unchanged"]
        and deterministic["causal_support_unchanged"]
        and deterministic["full_context_future_sensitive"]
        and random_weight["causal_scores_unchanged"]
        and random_weight["causal_support_unchanged"]
    )
    summary = {
        "status": "pass" if passed else "fail",
        "decision": (
            "causal_router_future_perturbation_invariant"
            if passed
            else "causal_router_future_perturbation_failed"
        ),
        "claim_status": (
            "contextual_mlp_causal_future_invariance_supported"
            if passed
            else "contextual_mlp_causal_future_invariance_blocked"
        ),
        "future_perturbation_invariance": passed,
        "checks": {
            "deterministic_contrast": deterministic,
            "random_weight_causal_router": random_weight,
        },
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    _write_artifacts(out_dir, summary)
    return summary


def _deterministic_contrast(torch: Any, residual_cls: Any) -> dict[str, Any]:
    hidden_dim = 4
    causal = residual_cls(
        hidden_dim=hidden_dim,
        num_columns=5,
        atoms_per_column=2,
        top_k=2,
        support_router="contextual_mlp_causal",
        contextual_router_hidden_dim=8,
    )
    full_context = residual_cls(
        hidden_dim=hidden_dim,
        num_columns=5,
        atoms_per_column=2,
        top_k=2,
        support_router="contextual_mlp",
        contextual_router_hidden_dim=8,
    )
    for residual in (causal, full_context):
        with torch.no_grad():
            first = residual.contextual_column_scores[1]
            second = residual.contextual_column_scores[3]
            first.weight.zero_()
            first.bias.zero_()
            second.weight.zero_()
            next_hidden_feature = hidden_dim * 2
            first.weight[0, next_hidden_feature] = 1.0
            second.weight[0, 0] = 1.0

    hidden = torch.zeros(1, 5, hidden_dim)
    hidden[:, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    perturbed = hidden.clone()
    perturb_start = 2
    perturbed[:, perturb_start:, 0] += 100.0

    causal_scores = causal._score_columns(hidden)
    causal_perturbed_scores = causal._score_columns(perturbed)
    _, causal_support = causal(hidden, return_support=True)
    _, causal_perturbed_support = causal(perturbed, return_support=True)
    earlier = slice(0, perturb_start)
    causal_max_score_delta = float(
        (causal_scores[:, earlier, :] - causal_perturbed_scores[:, earlier, :])
        .abs()
        .max()
        .item()
    )

    full_scores = full_context._score_columns(hidden)
    full_perturbed_scores = full_context._score_columns(perturbed)
    leak_position = perturb_start - 1
    full_context_predecessor_max_score_delta = float(
        (full_scores[:, leak_position, :] - full_perturbed_scores[:, leak_position, :])
        .abs()
        .max()
        .item()
    )
    return {
        "perturb_start": perturb_start,
        "checked_positions_before_perturbation": perturb_start,
        "causal_max_score_delta_before_perturbation": causal_max_score_delta,
        "causal_scores_unchanged": causal_max_score_delta <= 1e-8,
        "causal_support_unchanged": bool(
            torch.equal(
                causal_support[:, earlier, :],
                causal_perturbed_support[:, earlier, :],
            )
        ),
        "full_context_predecessor_max_score_delta": full_context_predecessor_max_score_delta,
        "full_context_future_sensitive": full_context_predecessor_max_score_delta > 1e-3,
    }


def _random_weight_check(torch: Any, residual_cls: Any) -> dict[str, Any]:
    residual = residual_cls(
        hidden_dim=4,
        num_columns=5,
        atoms_per_column=2,
        top_k=2,
        support_router="contextual_mlp_causal",
        contextual_router_hidden_dim=8,
    )
    hidden = torch.randn(2, 5, 4)
    perturbed = hidden.clone()
    perturb_start = 2
    perturbed[:, perturb_start:, :] += torch.randn_like(perturbed[:, perturb_start:, :]) * 10.0

    base_scores = residual._score_columns(hidden)
    perturbed_scores = residual._score_columns(perturbed)
    _, base_support = residual(hidden, return_support=True)
    _, perturbed_support = residual(perturbed, return_support=True)
    max_score_delta = float(
        (base_scores[:, :perturb_start, :] - perturbed_scores[:, :perturb_start, :])
        .abs()
        .max()
        .item()
    )
    return {
        "perturb_start": perturb_start,
        "checked_positions_before_perturbation": perturb_start,
        "causal_max_score_delta_before_perturbation": max_score_delta,
        "causal_scores_unchanged": max_score_delta <= 1e-8,
        "causal_support_unchanged": bool(
            torch.equal(
                base_support[:, :perturb_start, :],
                perturbed_support[:, :perturb_start, :],
            )
        ),
    }


def _failure_summary(*, out_dir: Path, start: float, reason: str) -> dict[str, Any]:
    return {
        "status": "fail",
        "decision": "causal_router_future_perturbation_unavailable",
        "claim_status": "contextual_mlp_causal_future_invariance_unevaluated",
        "future_perturbation_invariance": False,
        "failures": [{"field": "runtime", "reason": reason}],
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", summary)


def _write_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Causal Router Future Perturbation Evidence",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Future perturbation invariance: `{summary['future_perturbation_invariance']}`",
        "",
        "This artifact perturbs future hidden states and checks that earlier "
        "`contextual_mlp_causal` router scores and support selections are unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run_causal_router_future_perturbation(out_dir=args.out)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
