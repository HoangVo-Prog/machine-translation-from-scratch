"""
Factory functions referenced by the CLI via dotted-path strings.
Each factory receives the parsed args namespace and returns the
corresponding object.

Example CLI flags:
  --model_factory src.factories:build_model
  --tokenizer_factory src.factories:build_tokenizer
"""

from __future__ import annotations

import argparse

from src.data.dataset import SimpleDataLoader, build_dataloader
from src.data.tokenizer import TranslationTokenizer
from src.models.seq2seq import Seq2Seq


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(args: argparse.Namespace) -> TranslationTokenizer:
    """
    Train (if needed) and return a TranslationTokenizer.
    Caches BPE tokenizer models under args.tokenizer_cache_dir.
    """
    return TranslationTokenizer.train_and_cache(
        train_file=args.train_file,
        cache_dir=args.tokenizer_cache_dir,
        direction=args.direction,
        vocab_size=getattr(args, "vocab_size", 8000),
        shared_vocab=getattr(args, "shared_vocab", True),
        backend=getattr(args, "tokenizer_backend", "bpe"),
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_train_dataloader(args: argparse.Namespace, tokenizer: TranslationTokenizer) -> SimpleDataLoader:
    return build_dataloader(
        file_path=args.train_file,
        tokenizer=tokenizer,
        direction=args.direction,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 2),
        max_src_len=getattr(args, "max_src_len", 150),
        max_tgt_len=getattr(args, "max_tgt_len", 150),
    )


def build_eval_dataloader(args: argparse.Namespace, tokenizer: TranslationTokenizer) -> SimpleDataLoader:
    return build_dataloader(
        file_path=args.eval_file,
        tokenizer=tokenizer,
        direction=args.direction,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 2),
        max_src_len=getattr(args, "max_src_len", 150),
        max_tgt_len=getattr(args, "max_tgt_len", 150),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, tokenizer: TranslationTokenizer) -> Seq2Seq:
    """Build Seq2Seq model from CLI args."""
    src_vocab = tokenizer.src.vocab_size
    tgt_vocab = tokenizer.tgt.vocab_size

    model = Seq2Seq(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_dim=getattr(args, "embed_dim", 256),
        hidden_size=getattr(args, "hidden_size", 512),
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        attention_type=args.attention_type,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        src_pad_idx=tokenizer.src.pad_idx,
        tgt_pad_idx=tokenizer.tgt.pad_idx,
        sos_idx=tokenizer.tgt.bos_idx,
        eos_idx=tokenizer.tgt.eos_idx,
    )
    return model
