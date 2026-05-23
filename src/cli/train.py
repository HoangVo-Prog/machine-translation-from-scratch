"""
CLI entry point for training.

Usage:
    python -m src.cli.train \
        --model_factory src.factories:build_model \
        --train_dataloader_factory src.factories:build_train_dataloader \
        --eval_dataloader_factory src.factories:build_eval_dataloader \
        --tokenizer_factory src.factories:build_tokenizer \
        ...
"""

import argparse
import sys

import torch

from src.utils.misc import import_from_string, set_seed, count_parameters, get_device
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Seq2Seq MT model")

    # Factory dotted paths
    parser.add_argument("--model_factory", required=True)
    parser.add_argument("--train_dataloader_factory", required=True)
    parser.add_argument("--eval_dataloader_factory", required=True)
    parser.add_argument("--tokenizer_factory", required=True)

    # Data
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--direction", default="en2vi", help="e.g. en2vi or vi2en")
    parser.add_argument("--tokenizer_cache_dir", default="checkpoints/tokenizers")
    parser.add_argument("--tokenizer_backend", default="bpe", choices=["bpe"])
    parser.add_argument("--max_src_len", type=int, default=150)
    parser.add_argument("--max_tgt_len", type=int, default=150)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--shared_vocab", type=lambda x: x.lower() == "true", default=True)

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cell_type", default="lstm", choices=["rnn", "gru", "lstm"])
    parser.add_argument(
        "--attention_type",
        default="luong",
        type=lambda x: x.lower(),
        choices=["none", "bahdanau", "luong"],
    )
    parser.add_argument(
        "--bidirectional",
        type=lambda x: x.lower() == "true",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyper-params
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--metric_for_best_model", default="eval/bleu")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--optimizer_type", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", default=None)

    # Logging & checkpointing
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=500)

    # W&B
    parser.add_argument("--wandb_log_steps", type=int, default=500)
    parser.add_argument("--wandb_enabled", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--wandb_resume", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--wandb_project", default="mt-project")
    parser.add_argument("--wandb_run_name", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Direction: {args.direction}")

    # ------------------------------------------------------------------ #
    # 1. Tokenizer
    # ------------------------------------------------------------------ #
    print("Building tokenizer ...")
    tokenizer_fn = import_from_string(args.tokenizer_factory)
    tokenizer = tokenizer_fn(args)
    print(
        f"  src vocab: {tokenizer.src.vocab_size}  "
        f"tgt vocab: {tokenizer.tgt.vocab_size}"
    )

    # ------------------------------------------------------------------ #
    # 2. DataLoaders
    # ------------------------------------------------------------------ #
    print("Building dataloaders ...")
    train_loader_fn = import_from_string(args.train_dataloader_factory)
    eval_loader_fn = import_from_string(args.eval_dataloader_factory)

    train_loader = train_loader_fn(args, tokenizer)
    eval_loader = eval_loader_fn(args, tokenizer)
    print(f"  train batches: {len(train_loader)}  eval batches: {len(eval_loader)}")

    # ------------------------------------------------------------------ #
    # 3. Model
    # ------------------------------------------------------------------ #
    print("Building model ...")
    model_fn = import_from_string(args.model_factory)
    model = model_fn(args, tokenizer).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------ #
    # 4. Train
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        args=args,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()