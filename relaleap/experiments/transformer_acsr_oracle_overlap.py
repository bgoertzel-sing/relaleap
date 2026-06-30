"""Oracle-overlap targets for a redesigned Transformer-ACSR support objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path("results/reports/transformer_acsr_oracle_overlap")
DEFAULT_CONTEXT_KEYS = ("batch_index", "position_index", "token_index", "target_token")
LOSS_KEYS = ("support_loss", "loss", "ce_loss", "intervention_ce")
SUPPORT_KEYS = ("support", "support_pair", "support_set", "columns", "candidate_support")
PREFIX_SAFE_FEATURE_NAMES = (
    "position_index",
    "normalized_position",
    "learned_support_multihot",
)


def _coerce_support(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text[0] in "[(":
            loaded = json.loads(text.replace("(", "[").replace(")", "]"))
            return _coerce_support(loaded)
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    raise TypeError(f"cannot parse support value {value!r}")


def _row_support(row: dict[str, Any]) -> tuple[int, ...]:
    for key in SUPPORT_KEYS:
        if key in row and row[key] not in (None, ""):
            return _coerce_support(row[key])
    if "left_column" in row and "right_column" in row:
        return (int(row["left_column"]), int(row["right_column"]))
    if "column_left" in row and "column_right" in row:
        return (int(row["column_left"]), int(row["column_right"]))
    if "column" in row:
        return (int(row["column"]),)
    raise KeyError("support row is missing a support identifier")


def _row_loss(row: dict[str, Any]) -> float:
    for key in LOSS_KEYS:
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    raise KeyError("support row is missing a loss field")


def _context_key(row: dict[str, Any], context_keys: Iterable[str]) -> tuple[Any, ...]:
    keys = tuple(context_keys)
    present = tuple(row.get(key) for key in keys)
    if any(value is not None for value in present):
        return present
    return ("global_context",)


def build_soft_oracle_support_targets(
    rows: Iterable[dict[str, Any]],
    *,
    temperature: float = 0.05,
    context_keys: tuple[str, ...] = DEFAULT_CONTEXT_KEYS,
) -> list[dict[str, Any]]:
    """Convert exhaustive support-loss rows into regret-aware soft support targets.

    Lower-loss supports receive larger probabilities via ``softmax(-loss / temperature)``
    within each context. Oracle labels are training/evaluator teachers only; inference
    must stay prefix-safe and cannot consume these losses.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        support = _row_support(row)
        loss = _row_loss(row)
        context = _context_key(row, context_keys)
        grouped[context].append({"source": dict(row), "support": support, "loss": loss})

    targets: list[dict[str, Any]] = []
    for context, candidates in grouped.items():
        if not candidates:
            continue
        best_loss = min(candidate["loss"] for candidate in candidates)
        logits = [-(candidate["loss"] - best_loss) / temperature for candidate in candidates]
        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        normalizer = sum(weights)
        ranked = sorted(
            zip(candidates, weights, strict=True),
            key=lambda item: (item[0]["loss"], item[0]["support"]),
        )
        for rank, (candidate, weight) in enumerate(ranked, start=1):
            source = candidate["source"]
            output = {
                **{key: source.get(key) for key in context_keys if key in source},
                "support": json.dumps(list(candidate["support"])),
                "support_loss": candidate["loss"],
                "oracle_best_loss": best_loss,
                "oracle_regret": candidate["loss"] - best_loss,
                "oracle_rank": rank,
                "target_probability": weight / normalizer,
                "target_temperature": temperature,
            }
            targets.append(output)
    return targets


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _support_text(value: Any) -> str:
    return ",".join(str(item) for item in _coerce_support(value))


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _overlap(left: str, right: str) -> float:
    left_set = set(_coerce_support(left))
    right_set = set(_coerce_support(right))
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _oracle_summary_examples(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        oracle_support = row.get("oracle_support")
        if oracle_support in (None, ""):
            continue
        learned_support = row.get("learned_support", "")
        oracle_loss = _parse_optional_float(row.get("oracle_ce_loss"))
        learned_loss = _parse_optional_float(row.get("learned_ce_loss"))
        if oracle_loss is None or learned_loss is None:
            continue
        examples.append(
            {
                "episode_index": int(row.get("episode_index", 0) or 0),
                "position_index": int(row.get("position_index", 0) or 0),
                "learned_support": _support_text(learned_support),
                "oracle_support": _support_text(oracle_support),
                "oracle_ce_loss": oracle_loss,
                "learned_ce_loss": learned_loss,
                "best_singleton_support": _support_text(row.get("best_singleton_support", "")),
                "best_singleton_ce_loss": _parse_optional_float(row.get("best_singleton_ce_loss")),
                "best_pair_support": _support_text(row.get("best_pair_support", "")),
                "best_pair_ce_loss": _parse_optional_float(row.get("best_pair_ce_loss")),
                "best_one_swap_support": _support_text(row.get("best_one_swap_support", "")),
                "best_one_swap_ce_loss": _parse_optional_float(row.get("best_one_swap_ce_loss")),
                "oracle_regret": _parse_optional_float(row.get("oracle_regret")),
            }
        )
    return sorted(examples, key=lambda item: (item["episode_index"], item["position_index"]))


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    train_fraction: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) < 2:
        return examples, []
    episodes = sorted({example["episode_index"] for example in examples})
    if len(episodes) > 1:
        split_index = max(1, min(len(episodes) - 1, int(round(len(episodes) * train_fraction))))
        train_episodes = set(episodes[:split_index])
        train = [example for example in examples if example["episode_index"] in train_episodes]
        holdout = [example for example in examples if example["episode_index"] not in train_episodes]
        if train and holdout:
            return train, holdout
    split_index = max(1, min(len(examples) - 1, int(round(len(examples) * train_fraction))))
    return examples[:split_index], examples[split_index:]


def _candidate_loss_proxy(example: dict[str, Any], predicted_support: str) -> float:
    candidates = (
        ("oracle_support", "oracle_ce_loss"),
        ("learned_support", "learned_ce_loss"),
        ("best_singleton_support", "best_singleton_ce_loss"),
        ("best_pair_support", "best_pair_ce_loss"),
        ("best_one_swap_support", "best_one_swap_ce_loss"),
    )
    for support_key, loss_key in candidates:
        if predicted_support == example.get(support_key):
            loss = example.get(loss_key)
            if loss is not None:
                return float(loss)
    regret = float(example.get("oracle_regret") or 0.0)
    return float(example["learned_ce_loss"]) + max(0.0, regret)


def _modal_support(examples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example["oracle_support"]] += 1
    if not counts:
        return ""
    return min(counts, key=lambda support: (-counts[support], support))


def _position_support_lookup(examples: list[dict[str, Any]]) -> dict[int, str]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[example["position_index"]].append(example)
    return {position: _modal_support(group) for position, group in grouped.items()}


def _multihot_support(support: str, num_columns: int) -> list[float]:
    output = [0.0 for _ in range(num_columns)]
    for column in _coerce_support(support):
        if 0 <= column < num_columns:
            output[column] = 1.0
    return output


def _train_tiny_causal_transformer_policy(
    *,
    train_examples: list[dict[str, Any]],
    holdout_examples: list[dict[str, Any]],
    feature_mode: str,
    target_mode: str,
    training_steps: int,
    learning_rate: float,
    seed: int,
) -> list[str]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("torch is required for the local Transformer-ACSR pregate") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    support_labels = sorted({example["oracle_support"] for example in train_examples})
    if not support_labels:
        return ["" for _ in holdout_examples]
    label_to_index = {support: index for index, support in enumerate(support_labels)}
    max_position = max(1, max(example["position_index"] for example in train_examples + holdout_examples))
    max_column = 0
    for example in train_examples + holdout_examples:
        for key in ("learned_support", "oracle_support"):
            for column in _coerce_support(example[key]):
                max_column = max(max_column, int(column))
    num_columns = max_column + 1

    def features(example: dict[str, Any]) -> list[float]:
        position_feature = float(example["position_index"]) / float(max_position)
        if feature_mode == "token_position_only":
            return [position_feature]
        return [position_feature, 1.0] + _multihot_support(example["learned_support"], num_columns)

    def labels(examples: list[dict[str, Any]]) -> list[int]:
        raw = [example["oracle_support"] for example in examples]
        if target_mode == "shuffled":
            raw = list(reversed(raw))
        elif target_mode == "delayed":
            raw = raw[-1:] + raw[:-1]
        fallback = _modal_support(train_examples)
        return [label_to_index.get(support, label_to_index[fallback]) for support in raw]

    train_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in train_examples:
        train_by_episode[example["episode_index"]].append(example)
    train_sequences = [
        sorted(sequence, key=lambda example: example["position_index"])
        for _, sequence in sorted(train_by_episode.items())
    ]
    feature_dim = len(features(train_examples[0]))
    hidden_dim = max(8, min(32, feature_dim * 4))
    model = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=1,
        ),
        nn.Linear(hidden_dim, len(support_labels)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(max(1, training_steps)):
        total_loss = None
        for sequence in train_sequences:
            x = torch.tensor([features(example) for example in sequence], dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(labels(sequence), dtype=torch.long)
            length = x.shape[1]
            causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            encoded = model[0](x)
            encoded = model[1](encoded)
            encoded = model[2](encoded, mask=causal_mask)
            logits = model[3](encoded).squeeze(0)
            loss = F.cross_entropy(logits, y)
            total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            break
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    holdout_by_episode: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, example in enumerate(holdout_examples):
        holdout_by_episode[example["episode_index"]].append((index, example))
    predictions = [support_labels[0] for _ in holdout_examples]
    with torch.no_grad():
        for _, sequence_items in sorted(holdout_by_episode.items()):
            sequence_items = sorted(sequence_items, key=lambda item: item[1]["position_index"])
            x = torch.tensor([features(example) for _, example in sequence_items], dtype=torch.float32).unsqueeze(0)
            length = x.shape[1]
            causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            encoded = model[0](x)
            encoded = model[1](encoded)
            encoded = model[2](encoded, mask=causal_mask)
            logits = model[3](encoded).squeeze(0)
            predicted_indices = torch.argmax(logits, dim=-1).tolist()
            for (original_index, _), predicted_index in zip(sequence_items, predicted_indices, strict=True):
                predictions[original_index] = support_labels[int(predicted_index)]
    return predictions


def _policy_metric_row(
    *,
    policy_name: str,
    predictor_family: str,
    train_examples: list[dict[str, Any]],
    holdout_examples: list[dict[str, Any]],
    predictions: list[str],
) -> dict[str, Any]:
    losses = [_candidate_loss_proxy(example, prediction) for example, prediction in zip(holdout_examples, predictions, strict=True)]
    oracle_losses = [float(example["oracle_ce_loss"]) for example in holdout_examples]
    learned_losses = [float(example["learned_ce_loss"]) for example in holdout_examples]
    regrets = [loss - oracle for loss, oracle in zip(losses, oracle_losses, strict=True)]
    learned_regrets = [
        learned - oracle for learned, oracle in zip(learned_losses, oracle_losses, strict=True)
    ]
    exact = [
        prediction == example["oracle_support"]
        for example, prediction in zip(holdout_examples, predictions, strict=True)
    ]
    overlaps = [
        _overlap(prediction, example["oracle_support"])
        for example, prediction in zip(holdout_examples, predictions, strict=True)
    ]
    mean_loss = _mean(losses)
    mean_oracle = _mean(oracle_losses)
    mean_learned = _mean(learned_losses)
    mean_regret = _mean(regrets)
    mean_learned_regret = _mean(learned_regrets)
    return {
        "policy_name": policy_name,
        "predictor_family": predictor_family,
        "split": "heldout",
        "train_context_count": len(train_examples),
        "heldout_context_count": len(holdout_examples),
        "prefix_safe_feature_names": ";".join(PREFIX_SAFE_FEATURE_NAMES),
        "uses_target_token_as_predictor_feature": False,
        "uses_oracle_loss_as_predictor_feature": False,
        "mean_proxy_intervention_ce": mean_loss,
        "mean_oracle_ce": mean_oracle,
        "mean_learned_router_ce": mean_learned,
        "mean_oracle_regret": mean_regret,
        "mean_learned_router_regret": mean_learned_regret,
        "regret_recovery_fraction_vs_learned": (
            None
            if mean_learned_regret in (None, 0.0)
            else (mean_learned_regret - (mean_regret or 0.0)) / mean_learned_regret
        ),
        "oracle_exact_match_rate": sum(exact) / len(exact) if exact else None,
        "oracle_mean_jaccard_overlap": _mean(overlaps),
        "beats_learned_router_proxy_ce": (
            mean_loss is not None and mean_learned is not None and mean_loss <= mean_learned
        ),
        "beats_oracle_regret_null": False,
        "promotion_allowed": False,
        "requires_gpu_now": False,
        "advance_to_gpu_validation": False,
    }


def write_oracle_overlap_training_pregate(
    *,
    source_csv: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    training_steps: int = 40,
    learning_rate: float = 5e-3,
    seed: int = 17,
) -> dict[str, Any]:
    rows = _read_csv(source_csv)
    examples = _oracle_summary_examples(rows)
    train_examples, holdout_examples = _split_examples(examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    if train_examples and holdout_examples:
        primary_predictions = _train_tiny_causal_transformer_policy(
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            feature_mode="prefix_safe_router_state",
            target_mode="oracle",
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed,
        )
        token_position_predictions = _train_tiny_causal_transformer_policy(
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            feature_mode="token_position_only",
            target_mode="oracle",
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 1,
        )
        shuffled_predictions = _train_tiny_causal_transformer_policy(
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            feature_mode="prefix_safe_router_state",
            target_mode="shuffled",
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 2,
        )
        delayed_predictions = _train_tiny_causal_transformer_policy(
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            feature_mode="prefix_safe_router_state",
            target_mode="delayed",
            training_steps=training_steps,
            learning_rate=learning_rate,
            seed=seed + 3,
        )
        position_lookup = _position_support_lookup(train_examples)
        global_support = _modal_support(train_examples)
        frequency_predictions = [
            position_lookup.get(example["position_index"], global_support)
            for example in holdout_examples
        ]
        rng = random.Random(seed + 4)
        support_pool = sorted({example["oracle_support"] for example in train_examples})
        random_predictions = [
            rng.choice(support_pool) if support_pool else ""
            for _ in holdout_examples
        ]
        policies = (
            (
                "oracle_overlap_causal_transformer_support_predictor",
                "tiny_causal_transformer_prefix_safe_router_state",
                primary_predictions,
            ),
            (
                "token_position_transformer_null",
                "tiny_causal_transformer_token_position_only",
                token_position_predictions,
            ),
            (
                "shuffled_oracle_target_null",
                "tiny_causal_transformer_shuffled_targets",
                shuffled_predictions,
            ),
            (
                "delayed_oracle_target_null",
                "tiny_causal_transformer_delayed_targets",
                delayed_predictions,
            ),
            ("position_frequency_null", "position_frequency_lookup", frequency_predictions),
            ("random_frequency_null", "random_train_support_frequency", random_predictions),
        )
        metric_rows = [
            _policy_metric_row(
                policy_name=policy_name,
                predictor_family=predictor_family,
                train_examples=train_examples,
                holdout_examples=holdout_examples,
                predictions=predictions,
            )
            for policy_name, predictor_family, predictions in policies
        ]
        primary = metric_rows[0]
        null_rows = metric_rows[1:]
        primary_loss = primary["mean_proxy_intervention_ce"]
        primary_overlap = primary["oracle_mean_jaccard_overlap"]
        primary_regret = primary["mean_oracle_regret"]
        primary["beats_oracle_regret_null"] = all(
            primary_regret is not None
            and row["mean_oracle_regret"] is not None
            and primary_regret <= row["mean_oracle_regret"]
            for row in null_rows
        )
        primary["pregate_passes"] = (
            bool(primary["beats_learned_router_proxy_ce"])
            and bool(primary["beats_oracle_regret_null"])
            and primary_overlap is not None
            and primary_overlap >= 0.25
        )
        for row in null_rows:
            row["pregate_passes"] = False
            row["beats_oracle_regret_null"] = False
            row["primary_proxy_ce_margin"] = (
                None
                if primary_loss is None or row["mean_proxy_intervention_ce"] is None
                else row["mean_proxy_intervention_ce"] - primary_loss
            )
    _write_csv(out_dir / "oracle_overlap_training_pregate.csv", metric_rows)
    primary_row = metric_rows[0] if metric_rows else {}
    pregate_passes = bool(primary_row.get("pregate_passes"))
    summary = {
        "status": "pass" if metric_rows else "fail",
        "decision": (
            "oracle_overlap_transformer_acsr_training_pregate_passed_local_proxy"
            if pregate_passes
            else "oracle_overlap_transformer_acsr_training_pregate_gpu_blocked"
        ),
        "source_csv": str(source_csv),
        "source_format": "oracle_support_summary_rows",
        "source_row_count": len(rows),
        "usable_context_count": len(examples),
        "train_context_count": len(train_examples),
        "heldout_context_count": len(holdout_examples),
        "training_steps": training_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "primary_result": primary_row,
        "pregate_passes": pregate_passes,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": (
            "replace_proxy_row_pregate_with_hidden_feature_same_student_intervention_training"
            if not pregate_passes
            else "run_full_same_student_intervention_transformer_acsr_null_gate_locally"
        ),
        "artifacts": {
            "oracle_overlap_training_pregate_csv": str(
                out_dir / "oracle_overlap_training_pregate.csv"
            ),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    notes = [
        "# Transformer-ACSR Oracle-Overlap Training Pregate",
        "",
        f"- Source CSV: `{source_csv}`",
        f"- Usable contexts: `{len(examples)}`",
        f"- Train contexts: `{len(train_examples)}`",
        f"- Heldout contexts: `{len(holdout_examples)}`",
        f"- Decision: `{summary['decision']}`",
        f"- Next step: `{summary['selected_next_step']}`",
        "",
        "This is a local row-level pregate over same-student oracle-summary artifacts. "
        "Predictor features are prefix-safe row features only; evaluator targets and losses "
        "are used for training labels/evaluation, not inference features. GPU validation remains blocked.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def write_oracle_overlap_target_report(
    *,
    source_csv: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    temperature: float = 0.05,
) -> dict[str, Any]:
    rows = _read_csv(source_csv)
    targets = build_soft_oracle_support_targets(rows, temperature=temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "oracle_support_targets.csv"
    _write_csv(target_path, targets)
    contexts = {tuple(row.get(key) for key in DEFAULT_CONTEXT_KEYS) for row in targets}
    top_probability = max((row["target_probability"] for row in targets), default=None)
    summary = {
        "status": "pass",
        "decision": "oracle_overlap_transformer_acsr_target_construction_ready",
        "source_csv": str(source_csv),
        "candidate_row_count": len(rows),
        "target_row_count": len(targets),
        "context_count": len(contexts),
        "temperature": temperature,
        "max_target_probability": top_probability,
        "requires_gpu_now": False,
        "promotion_allowed": False,
        "advance_to_gpu_validation": False,
        "selected_next_step": (
            "wire_local_prefix_safe_transformer_training_with_oracle_overlap_and_null_gates"
        ),
        "artifacts": {
            "oracle_support_targets_csv": str(target_path),
            "summary_json": str(out_dir / "summary.json"),
            "notes_md": str(out_dir / "notes.md"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    notes = [
        "# Transformer-ACSR Oracle-Overlap Targets",
        "",
        f"- Source CSV: `{source_csv}`",
        f"- Candidate rows: `{len(rows)}`",
        f"- Target rows: `{len(targets)}`",
        f"- Temperature: `{temperature}`",
        "- GPU validation and promotion remain blocked; this is a local training-target primitive.",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--training-pregate",
        action="store_true",
        help="Train/evaluate the local oracle-overlap support predictor pregate.",
    )
    parser.add_argument("--training-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.training_pregate:
        summary = write_oracle_overlap_training_pregate(
            source_csv=args.source_csv,
            out_dir=args.out,
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        summary = write_oracle_overlap_target_report(
            source_csv=args.source_csv,
            out_dir=args.out,
            temperature=args.temperature,
        )
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
