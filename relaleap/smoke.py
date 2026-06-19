"""Phase 0 char-level smoke machinery for RelaLeap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TINY_SHAKESPEARE_EXCERPT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
""".strip()


@dataclass(frozen=True)
class Phase0Result:
    """Structured result returned by the Phase 0 smoke routine."""

    residual_objective: str
    vocab_size: int
    seq_len: int
    batch_size: int
    base_loss: float
    zero_init_loss: float
    initial_loss: float
    post_step_loss: float
    training_steps: int
    metric_rows: list[dict[str, float | int | str]]
    residual_parameter_delta: float
    max_zero_init_logit_delta: float
    max_hep_alpha0_logit_delta: float
    invariants: dict[str, bool]

    def to_summary(self) -> dict[str, Any]:
        return {
            "residual_objective": self.residual_objective,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "base_loss": self.base_loss,
            "zero_init_loss": self.zero_init_loss,
            "initial_loss": self.initial_loss,
            "post_step_loss": self.post_step_loss,
            "training_steps": self.training_steps,
            "residual_parameter_delta": self.residual_parameter_delta,
            "max_zero_init_logit_delta": self.max_zero_init_logit_delta,
            "max_hep_alpha0_logit_delta": self.max_hep_alpha0_logit_delta,
            "invariants": self.invariants,
        }

    def to_metric_rows(self) -> list[dict[str, float | int | str]]:
        """Return the real Phase 0 loss stream written to metrics.csv."""

        return self.metric_rows


def run_phase0_smoke(config: dict[str, Any]) -> Phase0Result:
    """Run local Phase 0 invariants against a tiny char transformer.

    The routine intentionally avoids dataset downloads. It uses a deterministic
    Shakespeare-shaped excerpt so the harness can run from a plain Python
    command in local CPU-only environments and temporary GPU backends.
    """

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Phase 0 smoke requires an importable torch install") from exc

    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    base_cfg = model_cfg.get("base", {})
    column_cfg = model_cfg.get("columns", {})
    inference_cfg = config.get("inference", {})
    training_cfg = config.get("training", {})

    seed = int(run_cfg.get("seed", 1))
    max_steps = int(run_cfg.get("max_steps", 10))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    residual_objective = str(
        training_cfg.get(
            "residual_objective",
            run_cfg.get("residual_objective", "supervised_ce"),
        )
    )
    seq_len = int(data_cfg.get("seq_len", 32))
    hidden_dim = int(base_cfg.get("hidden_dim", 32))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 8))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 1))
    pc_steps = int(inference_cfg.get("pc_steps", 1))
    hep_alpha = float(inference_cfg.get("hep_alpha", 0.0))

    if pc_steps < 1:
        raise ValueError("inference.pc_steps must be at least 1")
    if max_steps < 1:
        raise ValueError("run.max_steps must be at least 1 for Phase 0 smoke")
    if hep_alpha != 0.0:
        raise NotImplementedError("Phase 0 smoke only implements hep_alpha == 0.0")
    if residual_objective not in {"supervised_ce", "pc_logit_mse"}:
        raise ValueError(
            "training.residual_objective must be one of: supervised_ce, pc_logit_mse"
        )

    torch.manual_seed(seed)
    inputs, targets, vocab_size = _build_char_batch(seq_len=seq_len, batch_size=4)
    base = TinyCharTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    residual = ResidualColumns(
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
    )
    base.eval()
    residual.eval()

    with torch.no_grad():
        base_logits = base(inputs)
        zero_init_logits = base(inputs, residual_adapter=residual)
        hep_logits = forward_with_hep_alpha(
            base,
            residual,
            inputs,
            pc_steps=pc_steps,
            hep_alpha=0.0,
        )
        base_loss_tensor = F.cross_entropy(
            base_logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )
        zero_init_loss_tensor = F.cross_entropy(
            zero_init_logits[:, :-1, :].reshape(-1, vocab_size),
            targets[:, :-1].reshape(-1),
        )

    max_zero_delta = float((base_logits - zero_init_logits).abs().max().item())
    max_hep_delta = float((zero_init_logits - hep_logits).abs().max().item())

    base_snapshot = _clone_state_dict(base)
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    residual.train()
    before_residual = _clone_state_dict(residual)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=learning_rate)
    initial_loss_tensor = _residual_loss(
        base,
        residual,
        inputs,
        targets,
        vocab_size,
        objective=residual_objective,
    )
    metric_rows: list[dict[str, float | int | str]] = [
        _metric_row(
            step=0,
            phase="initial",
            residual_objective=residual_objective,
            base_loss=float(base_loss_tensor.detach().item()),
            residual_loss=float(initial_loss_tensor.detach().item()),
            zero_init_loss=float(zero_init_loss_tensor.detach().item()),
            residual_parameter_delta=0.0,
            max_zero_init_logit_delta=max_zero_delta,
            max_hep_alpha0_logit_delta=max_hep_delta,
        )
    ]
    post_step_loss_tensor = initial_loss_tensor
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _residual_loss(
            base,
            residual,
            inputs,
            targets,
            vocab_size,
            objective=residual_objective,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            post_step_loss_tensor = _residual_loss(
                base,
                residual,
                inputs,
                targets,
                vocab_size,
                objective=residual_objective,
            )
        metric_rows.append(
            _metric_row(
                step=step,
                phase=_residual_update_phase(residual_objective),
                residual_objective=residual_objective,
                base_loss=float(base_loss_tensor.detach().item()),
                residual_loss=float(post_step_loss_tensor.detach().item()),
                zero_init_loss=float(zero_init_loss_tensor.detach().item()),
                residual_parameter_delta=_state_dict_delta(before_residual, residual),
                max_zero_init_logit_delta=max_zero_delta,
                max_hep_alpha0_logit_delta=max_hep_delta,
            )
        )

    invariants = {
        "zero_init_identity": max_zero_delta <= 1e-6,
        "frozen_base_unchanged": _state_dict_equal(base_snapshot, base),
        "hep_alpha_0_equivalence": max_hep_delta <= 1e-6,
        "residual_parameters_updated": _state_dict_delta(before_residual, residual) > 0.0,
    }

    return Phase0Result(
        residual_objective=residual_objective,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=int(inputs.shape[0]),
        base_loss=float(base_loss_tensor.detach().item()),
        zero_init_loss=float(zero_init_loss_tensor.detach().item()),
        initial_loss=float(initial_loss_tensor.detach().item()),
        post_step_loss=float(post_step_loss_tensor.detach().item()),
        training_steps=max_steps,
        metric_rows=metric_rows,
        residual_parameter_delta=_state_dict_delta(before_residual, residual),
        max_zero_init_logit_delta=max_zero_delta,
        max_hep_alpha0_logit_delta=max_hep_delta,
        invariants=invariants,
    )


class TinyCharTransformer:
    """Small decoder-only-enough char model for smoke invariants."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch.nn as nn

        class _TinyCharTransformer(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                seq_len: int,
                hidden_dim: int,
                layers: int,
            ) -> None:
                super().__init__()
                import torch

                self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
                self.position_embedding = nn.Parameter(torch.zeros(seq_len, hidden_dim))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.norm = nn.LayerNorm(hidden_dim)
                self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

            def forward(self, input_ids: Any, residual_adapter: Any | None = None) -> Any:
                seq_len = int(input_ids.shape[1])
                hidden = self.token_embedding(input_ids) + self.position_embedding[:seq_len]
                hidden = self.encoder(hidden)
                hidden = self.norm(hidden)
                if residual_adapter is not None:
                    hidden = residual_adapter(hidden)
                return self.lm_head(hidden)

        return _TinyCharTransformer(*args, **kwargs)


class ResidualColumns:
    """Zero-initialized sparse column residual adapter.

    Each token selects top-k columns by score, averages that column support with
    learned atom weights, and adds a learned value vector. Atom values start at
    zero, so the adapter is exactly the identity at initialization while still
    receiving gradients during residual training.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _ResidualColumns(nn.Module):
            def __init__(
                self,
                hidden_dim: int,
                num_columns: int,
                atoms_per_column: int,
                top_k: int,
            ) -> None:
                super().__init__()
                if top_k < 1 or top_k > num_columns:
                    raise ValueError("top_k must be between 1 and num_columns")
                self.top_k = top_k
                self.column_scores = nn.Linear(hidden_dim, num_columns, bias=False)
                self.atom_logits = nn.Parameter(torch.zeros(num_columns, atoms_per_column))
                self.atom_values = nn.Parameter(
                    torch.zeros(num_columns, atoms_per_column, hidden_dim)
                )
                nn.init.zeros_(self.column_scores.weight)

            def forward(self, hidden: Any) -> Any:
                scores = self.column_scores(hidden)
                top_values, top_indices = scores.topk(self.top_k, dim=-1)
                column_weights = F.softmax(top_values, dim=-1)
                atom_weights = F.softmax(self.atom_logits, dim=-1)
                column_values = torch.einsum(
                    "ca,cah->ch",
                    atom_weights,
                    self.atom_values,
                )
                selected_values = column_values[top_indices]
                residual = torch.einsum("bsk,bskh->bsh", column_weights, selected_values)
                return hidden + residual

        return _ResidualColumns(*args, **kwargs)


def forward_with_hep_alpha(
    base: Any,
    residual: Any,
    input_ids: Any,
    *,
    pc_steps: int,
    hep_alpha: float,
) -> Any:
    """HEP inference shim used to pin alpha-0 equivalence."""

    if pc_steps < 1:
        raise ValueError("pc_steps must be at least 1")
    if hep_alpha != 0.0:
        raise NotImplementedError("HEP alpha > 0 is not implemented in Phase 0 smoke")
    return base(input_ids, residual_adapter=residual)


def _residual_loss(
    base: Any,
    residual: Any,
    inputs: Any,
    targets: Any,
    vocab_size: int,
    *,
    objective: str,
) -> Any:
    import torch
    import torch.nn.functional as F

    logits = base(inputs, residual_adapter=residual)
    prediction_logits = logits[:, :-1, :]
    prediction_targets = targets[:, :-1]
    if objective == "supervised_ce":
        return F.cross_entropy(
            prediction_logits.reshape(-1, vocab_size),
            prediction_targets.reshape(-1),
        )
    if objective == "pc_logit_mse":
        target_distribution = F.one_hot(
            prediction_targets,
            num_classes=vocab_size,
        ).to(dtype=prediction_logits.dtype)
        prediction_distribution = torch.softmax(prediction_logits, dim=-1)
        return F.mse_loss(prediction_distribution, target_distribution)
    raise ValueError(f"Unsupported residual objective: {objective}")


def _residual_update_phase(objective: str) -> str:
    if objective == "pc_logit_mse":
        return "pc_residual_update"
    return "residual_update"


def _metric_row(
    *,
    step: int,
    phase: str,
    residual_objective: str,
    base_loss: float,
    residual_loss: float,
    zero_init_loss: float,
    residual_parameter_delta: float,
    max_zero_init_logit_delta: float,
    max_hep_alpha0_logit_delta: float,
) -> dict[str, float | int | str]:
    return {
        "step": step,
        "phase": phase,
        "residual_objective": residual_objective,
        "base_loss": base_loss,
        "residual_loss": residual_loss,
        "zero_init_loss": zero_init_loss,
        "residual_parameter_delta": residual_parameter_delta,
        "max_zero_init_logit_delta": max_zero_init_logit_delta,
        "max_hep_alpha0_logit_delta": max_hep_alpha0_logit_delta,
    }


def _build_char_batch(seq_len: int, batch_size: int) -> tuple[Any, Any, int]:
    import torch

    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    text = (TINY_SHAKESPEARE_EXCERPT + "\n") * 16
    vocab = sorted(set(text))
    char_to_id = {char: index for index, char in enumerate(vocab)}
    encoded = torch.tensor([char_to_id[char] for char in text], dtype=torch.long)
    starts = torch.linspace(
        0,
        len(encoded) - seq_len - 2,
        steps=batch_size,
        dtype=torch.long,
    )
    rows = []
    targets = []
    for start in starts.tolist():
        rows.append(encoded[start : start + seq_len])
        targets.append(encoded[start + 1 : start + seq_len + 1])
    return torch.stack(rows), torch.stack(targets), len(vocab)


def _clone_state_dict(module: Any) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _state_dict_equal(snapshot: dict[str, Any], module: Any) -> bool:
    import torch

    return all(
        torch.equal(snapshot[key], value.detach())
        for key, value in module.state_dict().items()
    )


def _state_dict_delta(snapshot: dict[str, Any], module: Any) -> float:
    total = 0.0
    for key, value in module.state_dict().items():
        total += float((snapshot[key] - value.detach()).abs().sum().item())
    return total
