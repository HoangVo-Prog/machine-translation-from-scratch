"""
Trainer: handles training loop, evaluation, checkpointing, early stopping, W&B logging.
"""

import os
import math
import time
import shutil
from pathlib import Path
from typing import Optional, Callable

import torch
from tqdm import tqdm
import sacrebleu

from src.training.metrics import corpus_rouge_scores
from src.training.loss import LabelSmoothedCrossEntropy
from src.training.optimizers import build_optimizer, LinearWarmupDecayScheduler
from src.utils.tensor_ops import clip_grad_norm_manual


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        improved = (self.mode == "max" and value > self.best) or \
                   (self.mode == "min" and value < self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience   # True = stop


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        eval_loader,
        tokenizer,
        args,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.args = args
        self.device = device

        self.criterion = LabelSmoothedCrossEntropy(
            label_smoothing=args.label_smoothing,
            ignore_index=tokenizer.tgt.pad_idx,
        )

        # Optimizer
        self.optimizer = build_optimizer(
            optimizer_type=args.optimizer_type,
            params=model.parameters(),
            learning_rate=args.learning_rate,
        )

        # LR scheduler: linear warmup then linear decay
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = self._build_scheduler(total_steps, warmup_steps)

        self.global_step = 0
        self.best_metric = -float("inf")
        self.best_ckpt_path: Optional[str] = None
        self.start_epoch = 1

        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode="max")

        # Resume from checkpoint if provided
        resume_path = getattr(args, "resume_from_checkpoint", None)
        if resume_path:
            self._load_checkpoint(resume_path)

        # W&B
        self.wandb_run = None
        if args.wandb_enabled:
            self._init_wandb()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            eval_metrics = self._evaluate()

            bleu = eval_metrics["eval/bleu"]
            rouge1 = eval_metrics["eval/rouge1"]
            rouge2 = eval_metrics["eval/rouge2"]
            rouge_l = eval_metrics["eval/rougeL"]
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"eval_loss={eval_metrics['eval/loss']:.4f} | BLEU={bleu:.2f} | "
                f"ROUGE-1={rouge1:.2f} ROUGE-2={rouge2:.2f} ROUGE-L={rouge_l:.2f}"
            )

            # Save best
            target_metric = eval_metrics.get(self.args.metric_for_best_model, bleu)
            if target_metric > self.best_metric:
                self.best_metric = target_metric
                best_path = output_dir / "best_model.pt"
                self._save_checkpoint(str(best_path), epoch, eval_metrics)
                print(f"  ✓ New best model saved (metric={target_metric:.4f})")

            # Always save latest checkpoint for resuming
            last_path = output_dir / "last_checkpoint.pt"
            self._save_checkpoint(str(last_path), epoch, eval_metrics)

            # Early stopping
            if self.early_stopping.step(target_metric):
                print(f"Early stopping triggered after epoch {epoch}.")
                break

        print(f"\nTraining complete. Best {self.args.metric_for_best_model}={self.best_metric:.4f}")
        if self.wandb_run:
            self.wandb_run.finish()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str):
        print(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.start_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt["metrics"].get(
            self.args.metric_for_best_model, -float("inf")
        )

        # Restore early stopping state
        self.early_stopping.best = self.best_metric

        print(
            f"  Resumed at epoch {self.start_epoch} | "
            f"global_step={self.global_step} | "
            f"best_metric={self.best_metric:.4f}"
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for step, (src, tgt, src_lengths) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_lengths = src_lengths.to(self.device)

            logits = self.model(src, src_lengths, tgt, teacher_forcing_ratio=1.0)
            # logits: (batch, tgt_len-1, vocab)
            # tgt:    (batch, tgt_len)  -> targets are tgt[:, 1:]
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            (loss / self.args.gradient_accumulation_steps).backward()
            total_loss += loss.item()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                clip_grad_norm_manual(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.args.logging_steps == 0:
                    avg = total_loss / (step + 1)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

                if self.wandb_run and self.global_step % self.args.wandb_log_steps == 0:
                    avg = total_loss / (step + 1)
                    self.wandb_run.log(
                        {"train/loss": avg, "train/lr": self.scheduler.get_last_lr()[0]},
                        step=self.global_step,
                    )

                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    ckpt_path = Path(self.args.output_dir) / f"checkpoint-{self.global_step}.pt"
                    self._save_checkpoint(str(ckpt_path), epoch, {})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        hypotheses, references = [], []
        display_samples = []

        for src, tgt, src_lengths in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_lengths = src_lengths.to(self.device)

            # Loss
            logits = self.model(src, src_lengths, tgt, teacher_forcing_ratio=1.0)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1),
            )
            total_loss += loss.item()

            # BLEU: beam search decode
            best_seqs = self.model.beam_search(
                src, src_lengths, num_beams=self.args.num_beams
            )
            for seq, src_ids, ref_ids in zip(best_seqs, src.tolist(), tgt.tolist()):
                src_text = self.tokenizer.src.decode(src_ids)
                hyp = self.tokenizer.tgt.decode(seq)
                ref = self.tokenizer.tgt.decode(ref_ids)
                hypotheses.append(hyp)
                references.append([ref])
                if len(display_samples) < 3:
                    display_samples.append((src_text, ref, hyp))

        bleu = sacrebleu.corpus_bleu(hypotheses, references).score
        rouge_scores = corpus_rouge_scores(hypotheses, references)
        avg_loss = total_loss / len(self.eval_loader)

        # Print translation samples
        src_lang, tgt_lang = self.args.direction.split("2")
        print(f"\n  Translation samples:")
        for src_text, ref, hyp in display_samples:
            print(f"  [{src_lang}] {src_text}")
            print(f"  [{tgt_lang}] {ref}")
            print(f"  [pred] {hyp}")
            print()

        metrics = {
            "eval/loss": avg_loss,
            "eval/bleu": bleu,
            "eval/rouge1": rouge_scores["rouge1"],
            "eval/rouge2": rouge_scores["rouge2"],
            "eval/rougeL": rouge_scores["rougeL"],
        }
        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.global_step)
        return metrics

    def _save_checkpoint(self, path: str, epoch: int, metrics: dict):
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics,
                "args": vars(self.args),
            },
            path,
        )

    def _build_scheduler(self, total_steps: int, warmup_steps: int) -> LinearWarmupDecayScheduler:
        return LinearWarmupDecayScheduler(self.optimizer, total_steps, warmup_steps)

    def _init_wandb(self):
        try:
            import wandb

            resume = "allow" if self.args.wandb_resume else "never"
            self.wandb_run = wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name,
                resume=resume,
                config=vars(self.args),
            )
        except ImportError:
            print("wandb not installed — skipping W&B logging.")