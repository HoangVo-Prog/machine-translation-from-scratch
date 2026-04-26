"""Training loop for machine translation with optional W&B tracking."""

from __future__ import annotations

import copy
import csv
import logging
import os
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

try:
    from .evaluate import evaluate_model
    from .losses import maybe_compute_loss_from_outputs
    from .optimizer import build_optimizer, build_scheduler
except ImportError:  # pragma: no cover - fallback for direct script execution
    from evaluate import evaluate_model
    from losses import maybe_compute_loss_from_outputs
    from optimizer import build_optimizer, build_scheduler

LOGGER = logging.getLogger(__name__)
_DEPENDENCY_INSTALL_HINT = "pip install sacrebleu rouge-score wandb python-dotenv"
_DOTENV_LOADED = False


def get_secret(name: str, default: Any = None) -> Any:
    """Read a secret from Kaggle if available; return default otherwise."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        value = UserSecretsClient().get_secret(name)
        return default if value in (None, "") else value
    except Exception:
        return default


def _load_dotenv_if_available() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
        _DOTENV_LOADED = True
    except ImportError:
        LOGGER.warning("python-dotenv is not installed. Install with: %s", _DEPENDENCY_INSTALL_HINT)
        _DOTENV_LOADED = True


def resolve_env_var(name: str, default: Any = None) -> Any:
    """Resolve config/secret with priority: env > .env > Kaggle Secrets."""
    current = os.getenv(name)
    if current not in (None, ""):
        return current

    _load_dotenv_if_available()
    from_dotenv = os.getenv(name)
    if from_dotenv not in (None, ""):
        return from_dotenv

    from_secret = get_secret(name, default=None)
    if from_secret not in (None, ""):
        os.environ[name] = str(from_secret)
        return from_secret
    return default


def _cfg(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {}


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(x, device) for x in batch)
    if isinstance(batch, list):
        return [_move_to_device(x, device) for x in batch]
    return batch


def _extract_labels(batch: Any) -> torch.Tensor | None:
    if isinstance(batch, Mapping):
        for key in ("labels", "target_ids", "targets", "y"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
        return None
    if isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[1]):
        return batch[1]
    return None


def _forward_model(model: torch.nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            return model(batch[0], batch[1])
        return model(*batch)
    return model(batch)


def _init_wandb(config: Any) -> Any | None:
    enabled = bool(_cfg(config, "wandb_enabled", True))
    if not enabled:
        return None

    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb is not installed. Install with: %s. Continuing with W&B disabled.", _DEPENDENCY_INSTALL_HINT)
        return None

    api_key = resolve_env_var("WANDB_API_KEY", default=None)
    mode = str(_cfg(config, "wandb_mode", "online")).lower()
    if mode not in {"online", "offline", "disabled"}:
        mode = "online"

    if not api_key and mode == "online":
        LOGGER.warning("WANDB_API_KEY not found in env/.env/Kaggle; disabling W&B to avoid failure.")
        mode = "disabled"

    if api_key:
        try:
            wandb.login(key=api_key, relogin=False)
        except Exception as exc:
            LOGGER.warning("wandb login failed (%s). Falling back to disabled mode.", exc)
            mode = "disabled"

    if mode == "disabled":
        return None

    try:
        return wandb.init(
            project=_cfg(config, "wandb_project", "machine-translation"),
            entity=_cfg(config, "wandb_entity", None),
            name=_cfg(config, "wandb_run_name", None),
            group=_cfg(config, "wandb_group", None),
            tags=_cfg(config, "wandb_tags", None),
            mode=mode,
            config=_config_to_dict(config),
        )
    except Exception as exc:
        LOGGER.warning("wandb init failed (%s). Continuing with W&B disabled.", exc)
        return None


def _wandb_log(wandb_run: Any | None, data: dict[str, Any], step: int | None = None) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.log(data, step=step)
    except Exception as exc:
        LOGGER.warning("Failed to log to W&B: %s", exc)


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
    output_dir: Path,
    name: str,
    tokenizer: Any | None = None,
    extra_state: dict[str, Any] | None = None,
) -> str:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(checkpoint_dir))
    else:
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(checkpoint_dir))

    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, checkpoint_dir / "training_state.pt")
    return str(checkpoint_dir)


def _compute_total_training_steps(
    train_dataloader: Any, num_epochs: int, grad_accum_steps: int, max_steps: int | None
) -> int:
    if max_steps is not None and max_steps > 0:
        return int(max_steps)
    if not hasattr(train_dataloader, "__len__"):
        raise ValueError("train_dataloader must define __len__ if max_steps is not provided.")
    steps_per_epoch = max(1, len(train_dataloader) // max(1, grad_accum_steps))
    return steps_per_epoch * max(1, num_epochs)


def _decode_predictions(tokenizer: Any, sequences: Any, ignore_index: int = -100) -> list[str]:
    if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(sequences, skip_special_tokens=True)

    if torch.is_tensor(sequences):
        sequences = sequences.detach().cpu().tolist()
    if isinstance(sequences, (tuple, list)) and sequences and not isinstance(sequences[0], str):
        return [" ".join(str(int(tok)) for tok in seq if int(tok) != ignore_index) for seq in sequences]
    return [str(sequences)] if sequences is not None else []


def _append_translation_samples(csv_path: Path, rows: list[list[Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["epoch", "source", "reference", "prediction"])
        writer.writerows(rows)


def _get_fixed_eval_samples(eval_dataloader: Any, num_samples: int = 5) -> tuple[list[tuple[str, str]], torch.Tensor | None, torch.Tensor | None]:
    dataset = getattr(eval_dataloader, "dataset", None)
    if dataset is None:
        return [], None, None

    total = min(num_samples, len(dataset))
    if total == 0:
        return [], None, None

    raw_pairs = [(dataset.src_texts[i], dataset.trg_texts[i]) for i in range(total)]
    batch = None

    if hasattr(eval_dataloader, "collate_fn"):
        batch = eval_dataloader.collate_fn([dataset[i] for i in range(total)])

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return raw_pairs, batch[0], batch[1]

    samples = [dataset[i] for i in range(total)]
    if not samples:
        return [], None, None

    src_batch = torch.stack([item[0] for item in samples])
    tgt_batch = torch.stack([item[1] for item in samples])
    return raw_pairs, src_batch, tgt_batch


def _write_epoch_sample_translations(
    epoch: int,
    model: torch.nn.Module,
    eval_dataloader: Any,
    tokenizer: Any | None,
    output_dir: Path,
    generation_kwargs: dict[str, Any],
    num_samples: int = 5,
) -> None:
    raw_pairs, src_batch, _ = _get_fixed_eval_samples(eval_dataloader, num_samples=num_samples)
    if not raw_pairs or src_batch is None:
        return

    model.eval()
    with torch.no_grad():
        src_batch = src_batch.to(next(model.parameters()).device)
        dataset = getattr(eval_dataloader, "dataset", None)
        bos_token_id = None
        if dataset is not None and hasattr(dataset, "vocab_trg"):
            bos_token_id = dataset.vocab_trg.stoi.get("<sos>")
        if bos_token_id is None:
            bos_token_id = 1

        max_length = int(generation_kwargs.get("max_length", max(50, src_batch.shape[1] + 20)))
        try:
            pred_tokens = model.greedy_decode(
                src_batch,
                bos_token_id=bos_token_id,
                src_lengths=None,
                max_length=max_length,
            )
        except Exception:
            LOGGER.warning("Epoch %s: unable to generate translation samples; skipping sample logging.", epoch)
            return

    predictions = _decode_predictions(tokenizer, pred_tokens)
    rows: list[list[Any]] = []

    LOGGER.info("epoch=%s translation samples:", epoch)
    for index, ((source, reference), prediction) in enumerate(zip(raw_pairs, predictions), start=1):
        LOGGER.info("epoch=%s sample=%s src=%s", epoch, index, source)
        LOGGER.info("epoch=%s sample=%s ref=%s", epoch, index, reference)
        LOGGER.info("epoch=%s sample=%s pred=%s", epoch, index, prediction)
        rows.append([epoch, source, reference, prediction])

    csv_path = output_dir / "translation_samples.csv"
    _append_translation_samples(csv_path, rows)


def train(
    model: torch.nn.Module,
    train_dataloader: Any,
    eval_dataloader: Any | None = None,
    config: Any | None = None,
    tokenizer: Any | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train a MT model with periodic evaluation and checkpointing."""
    config = config or {}
    output_dir = Path(_cfg(config, "output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    active_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(active_device)

    num_epochs = int(_cfg(config, "num_epochs", _cfg(config, "epochs", 1)))
    max_steps = _cfg(config, "max_steps", None)
    grad_accum_steps = int(_cfg(config, "gradient_accumulation_steps", 1))
    logging_steps = int(_cfg(config, "logging_steps", 10))
    eval_steps = _cfg(config, "eval_steps", None)
    save_steps = _cfg(config, "save_steps", None)
    ignore_index = int(_cfg(config, "ignore_index", 0))
    label_smoothing = float(_cfg(config, "label_smoothing", 0.0))
    max_grad_norm = float(_cfg(config, "max_grad_norm", 1.0))
    metric_for_best = str(_cfg(config, "metric_for_best_model", "eval/bleu"))
    greater_is_better = bool(_cfg(config, "greater_is_better", True))
    generation_kwargs = copy.deepcopy(_cfg(config, "generation_kwargs", {}))
    wandb_log_steps = int(_cfg(config, "wandb_log_steps", 100))
    dropout = float(_cfg(config, "dropout", 0.1))
    optimizer_type = str(_cfg(config, "optimizer_type", "adamw"))
    early_stopping_patience = _cfg(config, "early_stopping_patience", None)
    min_lr = float(_cfg(config, "min_lr", 1e-6))

    # Merge generation kwargs from CLI
    if _cfg(config, "generation_temperature", None) is not None:
        generation_kwargs["temperature"] = float(_cfg(config, "generation_temperature"))
    if _cfg(config, "generation_top_k", None) is not None:
        generation_kwargs["top_k"] = int(_cfg(config, "generation_top_k"))
    if _cfg(config, "generation_top_p", None) is not None:
        generation_kwargs["top_p"] = float(_cfg(config, "generation_top_p"))
    if _cfg(config, "generation_repetition_penalty", None) is not None:
        generation_kwargs["repetition_penalty"] = float(_cfg(config, "generation_repetition_penalty"))

    total_steps = _compute_total_training_steps(train_dataloader, num_epochs, grad_accum_steps, max_steps)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)

    use_amp = bool(_cfg(config, "mixed_precision", False)) and active_device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    wandb_run = _init_wandb(config)

    best_metric = float("-inf") if greater_is_better else float("inf")
    best_checkpoint: str | None = None
    last_checkpoint: str | None = None
    global_step = 0
    early_stopping_counter = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader, start=1):
            batch = _move_to_device(batch, active_device)
            labels = _extract_labels(batch)

            autocast_context = autocast(device_type=active_device.type) if use_amp else nullcontext()
            with autocast_context:
                outputs = _forward_model(model, batch)
                loss, loss_logs = maybe_compute_loss_from_outputs(
                    outputs=outputs,
                    labels=labels,
                    label_smoothing=label_smoothing,
                    ignore_index=ignore_index,
                )

            scaled_loss = loss / max(1, grad_accum_steps)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % grad_accum_steps != 0:
                continue

            grad_norm = None
            if max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            train_logs = dict(loss_logs)
            train_logs["train/lr"] = float(optimizer.param_groups[0]["lr"])
            if grad_norm is not None:
                train_logs["train/grad_norm"] = grad_norm

            if global_step % logging_steps == 0:
                LOGGER.info("step=%s %s", global_step, train_logs)
            if wandb_run is not None and global_step % wandb_log_steps == 0:
                _wandb_log(wandb_run, train_logs, step=global_step)

            if eval_dataloader is not None and eval_steps and global_step % int(eval_steps) == 0:
                metrics = evaluate_model(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    tokenizer=tokenizer,
                    device=active_device,
                    generation_kwargs=generation_kwargs,
                    ignore_index=ignore_index,
                )
                LOGGER.info("eval step=%s metrics=%s", global_step, metrics)
                _wandb_log(wandb_run, metrics, step=global_step)

                current_metric = float(metrics.get(metric_for_best, float("nan")))
                if not torch.isnan(torch.tensor(current_metric)):
                    improved = current_metric > best_metric if greater_is_better else current_metric < best_metric
                    if improved:
                        best_metric = current_metric
                        early_stopping_counter = 0
                        best_checkpoint = _save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            output_dir=output_dir,
                            name="best",
                            tokenizer=tokenizer,
                            extra_state={
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_metric": best_metric,
                                "metric_for_best_model": metric_for_best,
                            },
                        )
                    else:
                        early_stopping_counter += 1
                        if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                            LOGGER.info("Early stopping triggered after %s epochs without improvement.", early_stopping_patience)
                            break

            if save_steps and global_step % int(save_steps) == 0:
                last_checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=output_dir,
                    name=f"step_{global_step}",
                    tokenizer=tokenizer,
                    extra_state={"epoch": epoch, "global_step": global_step},
                )

            if max_steps is not None and global_step >= int(max_steps):
                break

        if max_steps is not None and global_step >= int(max_steps):
            break

        if eval_dataloader is not None:
            _write_epoch_sample_translations(
                epoch=epoch + 1,
                model=model,
                eval_dataloader=eval_dataloader,
                tokenizer=tokenizer,
                output_dir=output_dir,
                generation_kwargs=generation_kwargs,
            )

    if eval_dataloader is not None and bool(_cfg(config, "run_eval_at_end", True)):
        final_metrics = evaluate_model(
            model=model,
            eval_dataloader=eval_dataloader,
            tokenizer=tokenizer,
            device=active_device,
            generation_kwargs=generation_kwargs,
            ignore_index=ignore_index,
        )
        LOGGER.info("final eval metrics=%s", final_metrics)
        _wandb_log(wandb_run, final_metrics, step=global_step)

    last_checkpoint = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        output_dir=output_dir,
        name="last",
        tokenizer=tokenizer,
        extra_state={"global_step": global_step, "best_metric": best_metric},
    )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            LOGGER.warning("W&B run finish failed.")

    return {
        "global_step": global_step,
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "metric_for_best_model": metric_for_best,
    }
