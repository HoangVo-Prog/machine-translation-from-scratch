"""
From-scratch BPE tokenizer with shared / separate vocab support.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class BPETokenizer:
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)

        self.vocab: List[str] = model["vocab"]
        self.token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = self.vocab
        self.merges: List[Tuple[str, str]] = [tuple(pair) for pair in model["merges"]]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        for word in text.strip().split():
            for token in self._encode_word(word):
                ids.append(self.token_to_id.get(token, UNK_IDX))

        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token[idx] for idx in ids if 0 <= idx < len(self.id_to_token)]
        tokens = [token for token in tokens if token not in SPECIAL_TOKENS]
        text = "".join(tokens)
        return text.replace("▁", " ").strip()

    def _encode_word(self, word: str) -> List[str]:
        if not word:
            return []
        symbols = [f"▁{word[0]}"] + list(word[1:])

        while len(symbols) > 1:
            best_pair = None
            best_rank = None
            for idx in range(len(symbols) - 1):
                pair = (symbols[idx], symbols[idx + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            symbols = _merge_pair_in_symbols(symbols, best_pair)
        return symbols

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def bos_idx(self) -> int:
        return BOS_IDX

    @property
    def eos_idx(self) -> int:
        return EOS_IDX

    @property
    def unk_idx(self) -> int:
        return UNK_IDX


class TranslationTokenizer:
    """
    Holds src and tgt tokenizers (may be the same object for shared vocab).
    """

    def __init__(self, src_tokenizer: BPETokenizer, tgt_tokenizer: BPETokenizer):
        self.src = src_tokenizer
        self.tgt = tgt_tokenizer

    @classmethod
    def from_cache(cls, cache_dir: str, direction: str) -> "TranslationTokenizer":
        cache_dir = Path(cache_dir)
        shared_path = cache_dir / "shared.bpe.json"
        if shared_path.exists():
            tok = BPETokenizer(str(shared_path))
            return cls(tok, tok)

        src_lang, tgt_lang = direction.split("2")
        src_path = cache_dir / f"{src_lang}.bpe.json"
        tgt_path = cache_dir / f"{tgt_lang}.bpe.json"
        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(
                f"BPE cache not found in {cache_dir}. Expected {src_path.name} and {tgt_path.name}."
            )
        return cls(BPETokenizer(str(src_path)), BPETokenizer(str(tgt_path)))

    @classmethod
    def train_and_cache(
        cls,
        train_file: str,
        cache_dir: str,
        direction: str,
        vocab_size: int = 8000,
        shared_vocab: bool = True,
        character_coverage: float = 1.0,
        backend: str = "bpe",
    ) -> "TranslationTokenizer":
        del character_coverage  # not used for this from-scratch BPE backend
        backend = backend.lower()
        if backend != "bpe":
            raise ValueError(f"Unsupported tokenizer backend='{backend}'. Only 'bpe' is available.")

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        src_lang, tgt_lang = direction.split("2")

        src_sentences, tgt_sentences = [], []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                src_sentences.append(obj[src_lang])
                tgt_sentences.append(obj[tgt_lang])

        if shared_vocab:
            shared_path = cache_dir / "shared.bpe.json"
            if not shared_path.exists():
                model = _train_bpe_model(src_sentences + tgt_sentences, vocab_size)
                _save_model(shared_path, model)
            tok = BPETokenizer(str(shared_path))
            return cls(tok, tok)

        src_path = cache_dir / f"{src_lang}.bpe.json"
        tgt_path = cache_dir / f"{tgt_lang}.bpe.json"
        if not src_path.exists():
            _save_model(src_path, _train_bpe_model(src_sentences, vocab_size))
        if not tgt_path.exists():
            _save_model(tgt_path, _train_bpe_model(tgt_sentences, vocab_size))
        return cls(BPETokenizer(str(src_path)), BPETokenizer(str(tgt_path)))


def _word_to_symbols(word: str) -> Tuple[str, ...]:
    if not word:
        return tuple()
    return tuple([f"▁{word[0]}"] + list(word[1:]))


def _merge_pair_in_symbols(symbols: List[str], pair: Tuple[str, str]) -> List[str]:
    merged: List[str] = []
    idx = 0
    while idx < len(symbols):
        if idx < len(symbols) - 1 and symbols[idx] == pair[0] and symbols[idx + 1] == pair[1]:
            merged.append(symbols[idx] + symbols[idx + 1])
            idx += 2
        else:
            merged.append(symbols[idx])
            idx += 1
    return merged


def _train_bpe_model(sentences: List[str], vocab_size: int) -> Dict:
    word_freq = Counter()
    for sentence in sentences:
        for word in sentence.strip().split():
            if word:
                word_freq[word] += 1

    words = {word: list(_word_to_symbols(word)) for word in word_freq}
    vocab_symbols = set()
    for symbols in words.values():
        vocab_symbols.update(symbols)

    target_vocab_size = max(vocab_size, len(SPECIAL_TOKENS) + len(vocab_symbols))
    merges: List[Tuple[str, str]] = []

    while len(SPECIAL_TOKENS) + len(vocab_symbols) < target_vocab_size:
        pair_freq = Counter()
        for word, symbols in words.items():
            freq = word_freq[word]
            for idx in range(len(symbols) - 1):
                pair_freq[(symbols[idx], symbols[idx + 1])] += freq
        if not pair_freq:
            break
        best_pair, best_count = pair_freq.most_common(1)[0]
        if best_count <= 0:
            break
        merges.append(best_pair)
        new_symbol = best_pair[0] + best_pair[1]
        vocab_symbols.add(new_symbol)
        for word, symbols in words.items():
            words[word] = _merge_pair_in_symbols(symbols, best_pair)

    ordered_vocab = SPECIAL_TOKENS + sorted(vocab_symbols)
    return {
        "version": "bpe_v1",
        "vocab": ordered_vocab,
        "merges": [list(pair) for pair in merges],
    }


def _save_model(path: Path, model: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
