"""
From-scratch dataset + dataloader style batching for IWSLT-style JSONL files.
"""

import json
import random
from math import ceil
from typing import List, Tuple

import torch

from src.data.tokenizer import TranslationTokenizer


class TranslationDataset:
    def __init__(
        self,
        file_path: str,
        tokenizer: TranslationTokenizer,
        direction: str,            # e.g. "en2vi" or "vi2en"
        max_src_len: int = 150,
        max_tgt_len: int = 150,
    ):
        self.tokenizer = tokenizer
        self.direction = direction
        self.src_lang, self.tgt_lang = direction.split("2")
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.data: List[Tuple[List[int], List[int]]] = []
        self._load(file_path)

    def _load(self, file_path: str):
        with open(file_path) as f:
            for line in f:
                obj = json.loads(line)
                src_text = obj[self.src_lang]
                tgt_text = obj[self.tgt_lang]

                src_ids = self.tokenizer.src.encode(src_text, add_bos=False, add_eos=True)
                tgt_ids = self.tokenizer.tgt.encode(tgt_text, add_bos=True, add_eos=True)

                if len(src_ids) > self.max_src_len or len(tgt_ids) > self.max_tgt_len:
                    continue

                self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(pad_idx: int):
    def _right_pad_sequences(sequences: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(seq.size(0) for seq in sequences)
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_idx,
            dtype=sequences[0].dtype,
        )
        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq
        return padded

    def _collate(batch):
        src_batch, tgt_batch = zip(*batch)
        src_lengths = torch.tensor([s.size(0) for s in src_batch])
        src_padded = _right_pad_sequences(list(src_batch))
        tgt_padded = _right_pad_sequences(list(tgt_batch))
        return src_padded, tgt_padded, src_lengths

    return _collate


class SimpleDataLoader:
    def __init__(
        self,
        dataset: TranslationDataset,
        batch_size: int,
        shuffle: bool,
        collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate = collate

    def __len__(self):
        if len(self.dataset) == 0:
            return 0
        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start: start + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate(batch)


def build_dataloader(
    file_path: str,
    tokenizer: TranslationTokenizer,
    direction: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 2,
    max_src_len: int = 150,
    max_tgt_len: int = 150,
) -> SimpleDataLoader:
    dataset = TranslationDataset(
        file_path, tokenizer, direction, max_src_len, max_tgt_len
    )
    _ = num_workers  # kept for API compatibility
    return SimpleDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate=collate_fn(tokenizer.src.pad_idx),
    )
