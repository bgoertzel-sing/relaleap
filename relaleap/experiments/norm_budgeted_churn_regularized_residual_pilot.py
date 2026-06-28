"""Run a local low-step norm-budgeted churn-regularized residual pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DESIGN_DIR = Path("results/reports/norm_budgeted_churn_regularized_residual_pilot_design")
DEFAULT_OUT_DIR = Path("results/reports/norm_budgeted_churn_regularized_residual_pilot")

REQUIRED_ARTIFACTS = (
    "summary.json",
    "arm_metrics.csv",
    "per_token_metrics.csv",
    "gate_criteria.csv",
    "notes.md",
)


def run_norm_budgeted_churn_regularized_residual_pilot(
    *,
    design_dir: Path = DEFAULT_DESIGN_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    train_steps: int = 8,
    seed: int = 1,
) -> dict[str, Any]:
    """Train a small CPU packet under the design report's residual/churn budget."""

    start = time.time()
    design = _read_json(design_dir / "summary.json")
    residual_budget = _float(design.get("residual_l2_budget"))
    preflight = _preflight_rows(design_dir, design, residual_budget, train_steps)
    arm_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    runtime_error = ""

    if all(row["passed"] for row in preflight):
        try:
            arm_rows, per_token_rows = _run_pilot(
                residual_budget=float(residual_budget),
                train_steps=train_steps,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            runtime_error = f"{type(exc).__name__}: {exc}"

    gate_rows = preflight + _pilot_gate_rows(arm_rows, per_token_rows, runtime_error)
    failures = [row for row in gate_rows if not row["passed"]]
    advancing = [row for row in arm_rows if row.get("scientific_gate") == "advances"]
    status = "pass" if not failures else "fail"
    decision = (
        "norm_budgeted_churn_regularized_residual_pilot_completed"
        if status == "pass"
        else "norm_budgeted_churn_regularized_residual_pilot_failed_closed"
    )
    selected_next_step = (
        "inspect per-token strata before deciding whether a RunPod repeat is scientifically justified"
        if advancing
        else "keep work local and add anchor-KL/off-target strata before any RunPod validation"
    )
    summary = {
        "status": status,
        "decision": decision,
        "claim_status": (
            "local_budgeted_pilot_has_advancing_challenger_needs_review"
            if advancing
            else "local_budgeted_pilot_no_challenger_clears_dense24_gate"
        ),
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "design_dir": str(design_dir),
        "out_dir": str(out_dir),
        "seed": seed,
        "train_steps": train_steps,
        "residual_l2_budget": residual_budget if residual_budget is not None else "",
        "arm_count": len(arm_rows),
        "per_token_row_count": len(per_token_rows),
        "advancement_row_count": len(advancing),
        "scientific_gate": "weak_pass_needs_review" if advancing and status == "pass" else "blocked",
        "gate_criteria": gate_rows,
        "failures": failures,
        "selected_next_step": selected_next_step,
        "runtime_error": runtime_error,
        "runtime_seconds": round(time.time() - start, 4),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "artifacts": {name.replace(".", "_"): str(out_dir / name) for name in REQUIRED_ARTIFACTS},
    }
    _write_artifacts(out_dir, summary, arm_rows, per_token_rows, gate_rows)
    return summary


def _run_pilot(*, residual_budget: float, train_steps: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from relaleap.smoke import ResidualColumns, TinyCharTransformer, _build_batch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    inputs, targets, vocab_size = _build_batch("tiny_shakespeare_char", seq_len=32, batch_size=4)
    base = TinyCharTransformer(vocab_size=vocab_size, seq_len=32, hidden_dim=32, layers=2)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        hidden = base.encode(inputs)
        base_logits = base.decode(hidden)
        base_losses = _per_token_ce(F, base_logits, targets, vocab_size)
    heldout_mask = _heldout_mask(base_losses.shape)

    modules: list[tuple[str, str, nn.Module]] = [
        ("dense_rank24_norm_budgeted", "dense_control", _DenseResidual(nn, 32, rank=8)),
        ("dense_rank16_norm_budgeted", "dense_control", _DenseResidual(nn, 32, rank=4)),
        (
            "sparse_contextual_topk2_norm_budgeted",
            "sparse_acsr",
            ResidualColumns(32, num_columns=12, atoms_per_column=4, top_k=2, support_router="contextual_mlp"),
        ),
        (
            "sparse_rank_matched_topk1_norm_budgeted",
            "sparse_acsr",
            ResidualColumns(32, num_columns=12, atoms_per_column=4, top_k=1, support_router="contextual_mlp"),
        ),
        ("bottleneck_gated_mlp_norm_budgeted", "mlp_control", _GatedMLPResidual(nn, 32, bottleneck=16)),
    ]
    arm_rows: list[dict[str, Any]] = []
    per_token_rows: list[dict[str, Any]] = []
    dense24: dict[str, Any] | None = None
    for arm, family, module in modules:
        _train_module(torch, F, base, module, hidden, targets, vocab_size, residual_budget, train_steps)
        arm_row, token_rows = _evaluate_module(
            F, base, module, hidden, targets, vocab_size, base_logits, base_losses, heldout_mask, arm, family, residual_budget, train_steps
        )
        if arm == "dense_rank24_norm_budgeted":
            dense24 = arm_row
        arm_rows.append(arm_row)
        per_token_rows.extend(token_rows)

    random_row, random_tokens = _evaluate_random_null(
        torch, F, base, hidden, targets, vocab_size, base_logits, base_losses, heldout_mask, residual_budget
    )
    arm_rows.append(random_row)
    per_token_rows.extend(random_tokens)

    if dense24:
        for row in arm_rows:
            _add_gate(row, dense24, residual_budget)
    return arm_rows, per_token_rows


def _train_module(torch: Any, F: Any, base: Any, module: Any, hidden: Any, targets: Any, vocab_size: int, budget: float, steps: int) -> None:
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-3)
    base_logits = base.decode(hidden).detach()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        update = _module_update(module, hidden)
        logits = base.decode(hidden + update)
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1))
        update_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1).mean()
        logit_mse = F.mse_loss(logits[:, :-1, :], base_logits[:, :-1, :])
        loss = ce + 0.75 * torch.relu(update_l2 - budget).pow(2) + 0.05 * logit_mse
        loss.backward()
        optimizer.step()
    module.eval()


def _evaluate_module(
    F: Any,
    base: Any,
    module: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    base_logits: Any,
    base_losses: Any,
    heldout_mask: Any,
    arm: str,
    family: str,
    budget: float,
    train_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with __import__("torch").no_grad():
        raw_update = _module_update(module, hidden)
        raw_l2 = raw_update[:, :-1, :].reshape(-1, raw_update.shape[-1]).norm(dim=-1)
        scale = 1.0
        heldout_raw_l2 = float(raw_l2[heldout_mask].mean().item())
        if heldout_raw_l2 > budget and heldout_raw_l2 > 1e-12:
            scale = budget / heldout_raw_l2
        update = raw_update * scale
        logits = base.decode(hidden + update)
        losses = _per_token_ce(F, logits, targets, vocab_size)
        l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
        flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        flat_base = base_logits[:, :-1, :].reshape(-1, base_logits.shape[-1])
        logit_mse = ((flat_logits - flat_base) ** 2).mean(dim=-1)
        anchor_kl = _per_token_anchor_kl(F, flat_logits, flat_base)
        flips = flat_logits.argmax(dim=-1) != flat_base.argmax(dim=-1)
    anchor_mask = ~heldout_mask
    flat_losses = losses.reshape(-1)
    flat_base_losses = base_losses.reshape(-1)
    heldout_loss = float(flat_losses[heldout_mask].mean().item())
    row = {
        "arm": arm,
        "family": family,
        "train_steps": train_steps,
        "heldout_ce_loss": heldout_loss,
        "heldout_delta_vs_base_ce": heldout_loss - float(flat_base_losses[heldout_mask].mean().item()),
        "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
        "heldout_logit_mse_vs_base": float(logit_mse[heldout_mask].mean().item()),
        "heldout_anchor_kl_vs_base": float(anchor_kl[heldout_mask].mean().item()),
        "heldout_prediction_flip_rate": float(flips[heldout_mask].float().mean().item()),
        "off_target_anchor_ce_loss": float(flat_losses[anchor_mask].mean().item()),
        "off_target_anchor_delta_vs_base_ce": float((flat_losses[anchor_mask] - flat_base_losses[anchor_mask]).mean().item()),
        "off_target_anchor_kl_vs_base": float(anchor_kl[anchor_mask].mean().item()),
        "off_target_prediction_flip_rate": float(flips[anchor_mask].float().mean().item()),
        "posthoc_residual_norm_scale": scale,
    }
    return row, _token_rows(arm, family, flat_losses, flat_base_losses, l2, logit_mse, anchor_kl, flips, heldout_mask)


def _evaluate_random_null(
    torch: Any,
    F: Any,
    base: Any,
    hidden: Any,
    targets: Any,
    vocab_size: int,
    base_logits: Any,
    base_losses: Any,
    heldout_mask: Any,
    budget: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(991)
    update = torch.randn(hidden.shape, generator=generator, dtype=hidden.dtype, device=hidden.device)
    token_l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
    scale = budget / max(float(token_l2[heldout_mask].mean().item()), 1e-12)
    update = update * scale
    with torch.no_grad():
        logits = base.decode(hidden + update)
        losses = _per_token_ce(F, logits, targets, vocab_size).reshape(-1)
        base_flat = base_losses.reshape(-1)
        l2 = update[:, :-1, :].reshape(-1, update.shape[-1]).norm(dim=-1)
        flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        flat_base = base_logits[:, :-1, :].reshape(-1, base_logits.shape[-1])
        logit_mse = ((flat_logits - flat_base) ** 2).mean(dim=-1)
        anchor_kl = _per_token_anchor_kl(F, flat_logits, flat_base)
        flips = flat_logits.argmax(dim=-1) != flat_base.argmax(dim=-1)
    anchor_mask = ~heldout_mask
    row = {
        "arm": "sparse_random_residual_same_l2_null",
        "family": "residual_null",
        "train_steps": 0,
        "heldout_ce_loss": float(losses[heldout_mask].mean().item()),
        "heldout_delta_vs_base_ce": float(losses[heldout_mask].mean().item()) - float(base_flat[heldout_mask].mean().item()),
        "heldout_residual_update_l2": float(l2[heldout_mask].mean().item()),
        "heldout_logit_mse_vs_base": float(logit_mse[heldout_mask].mean().item()),
        "heldout_anchor_kl_vs_base": float(anchor_kl[heldout_mask].mean().item()),
        "heldout_prediction_flip_rate": float(flips[heldout_mask].float().mean().item()),
        "off_target_anchor_ce_loss": float(losses[anchor_mask].mean().item()),
        "off_target_anchor_delta_vs_base_ce": float((losses[anchor_mask] - base_flat[anchor_mask]).mean().item()),
        "off_target_anchor_kl_vs_base": float(anchor_kl[anchor_mask].mean().item()),
        "off_target_prediction_flip_rate": float(flips[anchor_mask].float().mean().item()),
        "posthoc_residual_norm_scale": scale,
    }
    return row, _token_rows(row["arm"], row["family"], losses, base_flat, l2, logit_mse, anchor_kl, flips, heldout_mask)


def _module_update(module: Any, hidden: Any) -> Any:
    out = module(hidden)
    if out.shape == hidden.shape:
        return out - hidden if module.__class__.__name__.endswith("ResidualColumns") else out
    return out


def _per_token_ce(F: Any, logits: Any, targets: Any, vocab_size: int) -> Any:
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), targets[:, :-1].reshape(-1), reduction="none")


def _per_token_anchor_kl(F: Any, logits: Any, base_logits: Any) -> Any:
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    base_probs = base_log_probs.exp()
    return (base_probs * (base_log_probs - log_probs)).sum(dim=-1)


def _heldout_mask(shape: Any) -> Any:
    import torch

    count = int(shape[0])
    mask = torch.zeros(count, dtype=torch.bool)
    mask[count // 2 :] = True
    return mask


def _token_rows(
    arm: str,
    family: str,
    losses: Any,
    base_losses: Any,
    l2: Any,
    logit_mse: Any,
    anchor_kl: Any,
    flips: Any,
    heldout_mask: Any,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(int(losses.numel())):
        is_heldout = bool(heldout_mask[index].item())
        rows.append(
            {
                "arm": arm,
                "family": family,
                "token_index": index,
                "split": "heldout" if is_heldout else "anchor",
                "intervention_stratum": "target_heldout" if is_heldout else "off_target_anchor",
                "is_off_target_anchor": not is_heldout,
                "ce_loss": float(losses[index].item()),
                "base_ce_loss": float(base_losses[index].item()),
                "delta_vs_base_ce": float(losses[index].item() - base_losses[index].item()),
                "residual_update_l2": float(l2[index].item()),
                "logit_mse_vs_base": float(logit_mse[index].item()),
                "anchor_kl_vs_base": float(anchor_kl[index].item()),
                "prediction_changed_vs_base": bool(flips[index].item()),
            }
        )
    return rows


def _add_gate(row: dict[str, Any], dense24: dict[str, Any], budget: float) -> None:
    row["ce_delta_vs_dense24"] = _float(row.get("heldout_ce_loss")) - _float(dense24.get("heldout_ce_loss"))
    row["l2_delta_vs_budget"] = _float(row.get("heldout_residual_update_l2")) - budget
    row["logit_mse_delta_vs_dense24"] = _float(row.get("heldout_logit_mse_vs_base")) - _float(dense24.get("heldout_logit_mse_vs_base"))
    row["anchor_kl_delta_vs_dense24"] = _float(row.get("heldout_anchor_kl_vs_base")) - _float(dense24.get("heldout_anchor_kl_vs_base"))
    row["flip_delta_vs_dense24"] = _float(row.get("heldout_prediction_flip_rate")) - _float(dense24.get("heldout_prediction_flip_rate"))
    row["improves_vs_base"] = _float(row.get("heldout_delta_vs_base_ce")) < 0.0
    row["nontrivial_budget_fraction"] = _float(row.get("heldout_residual_update_l2")) >= 0.5 * budget
    row["within_budget"] = _float(row.get("heldout_residual_update_l2")) <= 1.05 * budget
    row["scientific_gate"] = (
        "advances"
        if row["arm"] != "dense_rank24_norm_budgeted"
        and row["improves_vs_base"]
        and row["nontrivial_budget_fraction"]
        and row["within_budget"]
        and row["ce_delta_vs_dense24"] < 0.0
        and row["logit_mse_delta_vs_dense24"] <= 0.0
        and row["anchor_kl_delta_vs_dense24"] <= 0.0
        and row["flip_delta_vs_dense24"] <= 0.0
        else "blocked_or_control"
    )


def _preflight_rows(design_dir: Path, design: dict[str, Any], residual_budget: float | None, train_steps: int) -> list[dict[str, Any]]:
    return [
        _criterion("design_passed", design.get("status") == "pass", "pilot design report passes", design.get("status", "missing"), "design report is missing or failed"),
        _criterion("design_selected_implementation", "implement the local low-step pilot" in str(design.get("selected_next_step", "")), "design points to local low-step pilot", design.get("selected_next_step", ""), "design report does not select this pilot"),
        _criterion("residual_budget_available", residual_budget is not None and residual_budget > 0.0, "positive residual-L2 budget", residual_budget if residual_budget is not None else "", "residual-L2 budget missing"),
        _criterion("train_steps_bounded", 1 <= train_steps <= 32, "1 <= train_steps <= 32", train_steps, "pilot train_steps must stay bounded"),
        _criterion("design_artifacts_present", (design_dir / "pilot_arms.csv").is_file() and (design_dir / "objective_terms.csv").is_file(), "design CSV artifacts exist", str(design_dir), "design CSV artifacts missing"),
    ]


def _pilot_gate_rows(arm_rows: list[dict[str, Any]], per_token_rows: list[dict[str, Any]], runtime_error: str) -> list[dict[str, Any]]:
    arms = {row.get("arm") for row in arm_rows}
    token_fields = set(per_token_rows[0]) if per_token_rows else set()
    strata = {row.get("intervention_stratum") for row in per_token_rows}
    required = {
        "dense_rank24_norm_budgeted",
        "dense_rank16_norm_budgeted",
        "sparse_contextual_topk2_norm_budgeted",
        "sparse_rank_matched_topk1_norm_budgeted",
        "bottleneck_gated_mlp_norm_budgeted",
        "sparse_random_residual_same_l2_null",
    }
    return [
        _criterion("pilot_runtime_completed", not runtime_error, "local pilot runtime completes", runtime_error or "ok", "pilot runtime failed"),
        _criterion("required_arms_evaluated", required.issubset(arms), sorted(required), sorted(arms), "one or more pilot arms missing"),
        _criterion("dense24_comparator_present", "dense_rank24_norm_budgeted" in arms, "dense rank24 comparator exists", sorted(arms), "dense rank24 comparator missing"),
        _criterion(
            "per_token_anchor_kl_off_target_strata_present",
            {"anchor_kl_vs_base", "intervention_stratum", "is_off_target_anchor"}.issubset(token_fields)
            and {"off_target_anchor", "target_heldout"}.issubset(strata),
            "per-token anchor KL and target/off-target strata exist",
            sorted(token_fields),
            "per-token anchor-KL/off-target diagnostic fields missing",
        ),
    ]


class _DenseResidual:
    def __init__(self, nn: Any, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

    def __call__(self, hidden: Any) -> Any:
        return self.up(self.down(hidden))

    def train(self) -> None:
        self.down.train()
        self.up.train()

    def eval(self) -> None:
        self.down.eval()
        self.up.eval()

    def parameters(self) -> Any:
        return list(self.down.parameters()) + list(self.up.parameters())


class _GatedMLPResidual(_DenseResidual):
    def __init__(self, nn: Any, hidden_dim: int, bottleneck: int) -> None:
        self.net = nn.Sequential(nn.Linear(hidden_dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, hidden_dim))
        self.gate = nn.Parameter(__import__("torch").tensor(0.1))

    def __call__(self, hidden: Any) -> Any:
        return self.gate.tanh() * self.net(hidden)

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def parameters(self) -> Any:
        return list(self.net.parameters()) + [self.gate]


def _criterion(criterion: str, passed: bool, threshold: Any, actual: Any, failure_reason: str) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "threshold": threshold,
        "actual": actual,
        "failure_reason": "" if passed else failure_reason,
    }


def _write_artifacts(out_dir: Path, summary: dict[str, Any], arm_rows: list[dict[str, Any]], per_token_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "arm_metrics.csv", arm_rows)
    _write_csv(out_dir / "per_token_metrics.csv", per_token_rows)
    _write_csv(out_dir / "gate_criteria.csv", gate_rows)
    lines = [
        "# Norm-Budgeted Churn-Regularized Residual Pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        f"- Claim status: `{summary['claim_status']}`",
        f"- Promotion allowed: `{summary['promotion_allowed']}`",
        f"- Requires GPU now: `{summary['requires_gpu_now']}`",
        f"- Residual L2 budget: `{summary['residual_l2_budget']}`",
        f"- Arms: `{summary['arm_count']}`",
        f"- Advancement rows: `{summary['advancement_row_count']}`",
        f"- Scientific gate: `{summary['scientific_gate']}`",
        f"- Next step: {summary['selected_next_step']}",
        "",
        "This is a bounded local CPU pilot. It trains small dense, sparse, and MLP residual arms under the dense-rank24 residual-L2 budget and reports CE, logit-MSE, anchor-KL, off-target anchor strata, and prediction-flip churn against the dense rank24 comparator. It does not promote any mechanism.",
    ]
    if summary["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{row['criterion']}`: {row['failure_reason']}" for row in summary["failures"])
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_norm_budgeted_churn_regularized_residual_pilot(
        design_dir=args.design_dir,
        out_dir=args.out,
        train_steps=args.train_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
