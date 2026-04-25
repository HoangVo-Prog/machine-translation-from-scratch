"""CLI command for machine translation training."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def _load_callable(path: str):
    """Load a callable from 'module.submodule:function_name'."""
    if ":" not in path:
        raise ValueError(
            f"Invalid callable path '{path}'. Use format: module.submodule:function_name"
        )
    if path.startswith("your_package."):
        raise ValueError(
            "Placeholder factory path detected. Replace 'your_package' with your real module path, "
            "for example: src.my_factories:build_model"
        )
    module_name, fn_name = path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Cannot import module '{module_name}' from '{path}'. "
            "Use a real factory path like 'src.my_factories:build_model'."
        ) from exc
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"'{fn_name}' in module '{module_name}' is not callable.")
    return fn


def _parse_config_value(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Use key=value.")
        key, val = item.split("=", 1)
        config[key.strip()] = _parse_config_value(val)
    return config


def _load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object/dict at the top level.")
    return data


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--scheduler_type", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default=None)
    parser.add_argument("--greater_is_better", type=str, default=None)
    parser.add_argument("--wandb_enabled", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)


def _read_bool_like(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MT model with src.training.trainer.train. "
            "This CLI is architecture-agnostic via factory callables."
        )
    )
    parser.add_argument(
        "--model_factory",
        type=str,
        required=True,
        help="Factory path: module.submodule:function_name (returns model).",
    )
    parser.add_argument(
        "--train_dataloader_factory",
        type=str,
        required=True,
        help="Factory path: module.submodule:function_name (returns train DataLoader).",
    )
    parser.add_argument(
        "--eval_dataloader_factory",
        type=str,
        default=None,
        help="Factory path: module.submodule:function_name (returns eval DataLoader).",
    )
    parser.add_argument(
        "--tokenizer_factory",
        type=str,
        default=None,
        help="Factory path: module.submodule:function_name (returns tokenizer).",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="Optional JSON config file path.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Extra config overrides as key=value.",
    )
    _add_common_flags(parser)
    return parser


def _merge_cli_flags_into_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)

    direct_keys = [
        "train_file",
        "eval_file",
        "output_dir",
        "num_epochs",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "gradient_accumulation_steps",
        "label_smoothing",
        "warmup_steps",
        "warmup_ratio",
        "scheduler_type",
        "max_grad_norm",
        "eval_steps",
        "save_steps",
        "logging_steps",
        "metric_for_best_model",
        "wandb_mode",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
    ]
    for key in direct_keys:
        value = getattr(args, key)
        if value is not None:
            merged[key] = value

    for key in ("mixed_precision", "greater_is_better", "wandb_enabled"):
        raw = getattr(args, key)
        parsed = _read_bool_like(raw) if raw is not None else None
        if parsed is not None:
            merged[key] = parsed

    merged.update(_parse_overrides(args.overrides))
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from src.training.trainer import train

    config_from_json = _load_json_config(args.config_json)
    config = _merge_cli_flags_into_config(args, config_from_json)

    try:
        model = _load_callable(args.model_factory)()
        train_dataloader = _load_callable(args.train_dataloader_factory)()
        eval_dataloader = (
            _load_callable(args.eval_dataloader_factory)() if args.eval_dataloader_factory else None
        )
        tokenizer = _load_callable(args.tokenizer_factory)() if args.tokenizer_factory else None
    except Exception as exc:
        raise ValueError(
            f"Factory loading failed: {exc}\n"
            "Tip: '--model_factory', '--train_dataloader_factory', '--eval_dataloader_factory', "
            "and '--tokenizer_factory' must point to real callables in your codebase."
        ) from exc

    result = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        tokenizer=tokenizer,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
