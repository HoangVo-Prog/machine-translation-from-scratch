import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.models.attentions import BahdanauAttention

from src.data.dataset import TranslationDataset
from src.data.vocab import Vocabulary
from src.data.vi_tokenizer import tokenize_vi
from src.data.en_tokenizer import EnglishBPETokenizer

from pathlib import Path


_CACHE = {}


def _build_data():
    base_dir = Path(__file__).resolve().parent.parent / "data"

    en_path = base_dir / "en_sents"
    vi_path = base_dir / "vi_sents"

    with open(en_path, "r", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f if line.strip()]

    with open(vi_path, "r", encoding="utf-8") as f:
        trg_texts = [line.strip() for line in f if line.strip()]

    if len(src_texts) != len(trg_texts):
        raise ValueError(
            f"Số câu EN và VI không khớp: {len(src_texts)} vs {len(trg_texts)}"
        )

    return src_texts, trg_texts


def _get_shared_objects():
    if "objects" in _CACHE:
        return _CACHE["objects"]

    src_texts, trg_texts = _build_data()

    tokenizer = EnglishBPETokenizer(num_merges=50)
    tokenizer.train(src_texts)

    vocab_src = Vocabulary(freq_threshold=1)
    vocab_trg = Vocabulary(freq_threshold=1)

    src_tokens = [
        ["<sos>"] + tokenizer.encode(text) + ["<eos>"]
        for text in src_texts
    ]

    trg_tokens = [
        ["<sos>"] + tokenize_vi(text) + ["<eos>"]
        for text in trg_texts
    ]

    vocab_src.build_vocabulary(src_tokens)
    vocab_trg.build_vocabulary(trg_tokens)

    _CACHE["objects"] = tokenizer, vocab_src, vocab_trg
    return _CACHE["objects"]


class TrainCollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch = [
            torch.tensor(item[0], dtype=torch.long)
            for item in batch
        ]

        trg_batch = [
            torch.tensor(item[1], dtype=torch.long)
            for item in batch
        ]

        src_padded = pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=self.pad_idx,
        )

        trg_padded = pad_sequence(
            trg_batch,
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return src_padded, trg_padded


def build_tokenizer():
    tokenizer, _, _ = _get_shared_objects()
    return tokenizer


def build_train_dataloader():
    src_texts, trg_texts = _build_data()
    tokenizer, vocab_src, vocab_trg = _get_shared_objects()

    dataset = TranslationDataset(
        src_texts,
        trg_texts,
        vocab_src,
        vocab_trg,
        tokenizer.encode,
        tokenize_vi,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=TrainCollateBatch(
            pad_idx=vocab_src.stoi["<pad>"]
        ),
    )


def build_eval_dataloader():
    src_texts, trg_texts = _build_data()
    tokenizer, vocab_src, vocab_trg = _get_shared_objects()

    dataset = TranslationDataset(
        src_texts,
        trg_texts,
        vocab_src,
        vocab_trg,
        tokenizer.encode,
        tokenize_vi,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=TrainCollateBatch(
            pad_idx=vocab_src.stoi["<pad>"]
        ),
    )


def build_model():
    _, vocab_src, vocab_trg = _get_shared_objects()

    hidden_size = 64
    embed_dim = 64

    encoder = Encoder(
        vocab_size=len(vocab_src),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=1,
        cell_type="gru",
    )

    attention = BahdanauAttention(
        encoder_hidden_dim=hidden_size,
        decoder_hidden_dim=hidden_size,
        attention_dim=32,
    )

    decoder = Decoder(
        vocab_size=len(vocab_trg),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=1,
        cell_type="gru",
        attention=attention,
        eos_token_id=vocab_trg.stoi["<eos>"],
    )

    return Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=vocab_src.stoi["<pad>"],
    )