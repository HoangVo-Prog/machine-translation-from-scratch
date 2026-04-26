import json
from pathlib import Path
import torch

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.models.attentions import BahdanauAttention

from src.data.dataset import get_dataloader
from src.data.vocab import Vocabulary
from src.data.vi_tokenizer import tokenize_vi
from src.data.en_tokenizer import EnglishBPETokenizer


_CACHE = {}


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_jsonl_parallel(path: Path):
    src_texts = []
    trg_texts = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            src = (
                obj.get("src")
                or obj.get("source")
                or obj.get("en")
                or obj.get("english")
                or obj.get("input")
            )

            trg = (
                obj.get("trg")
                or obj.get("target")
                or obj.get("vi")
                or obj.get("vietnamese")
                or obj.get("output")
            )

            if src is None or trg is None:
                raise ValueError(
                    f"Không tìm thấy cặp source/target trong dòng JSONL: {obj.keys()}"
                )

            src_texts.append(str(src).strip())
            trg_texts.append(str(trg).strip())

    return src_texts, trg_texts


def _read_raw_parallel(en_path: Path, vi_path: Path):
    with en_path.open("r", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f if line.strip()]

    with vi_path.open("r", encoding="utf-8") as f:
        trg_texts = [line.strip() for line in f if line.strip()]

    if len(src_texts) != len(trg_texts):
        raise ValueError(f"Số câu EN và VI không khớp: {len(src_texts)} vs {len(trg_texts)}")

    return src_texts, trg_texts


def _build_data(config=None, split="train"):
    root = _repo_root()

    file_key = "train_file" if split == "train" else "eval_file"
    file_path = _cfg(config, file_key, None)

    if file_path:
        path = root / file_path
        if not path.exists():
            path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {file_key}: {file_path}")

        if path.suffix == ".jsonl":
            return _read_jsonl_parallel(path)

        raise ValueError(f"Hiện chỉ hỗ trợ JSONL cho {file_key}: {path}")

    data_dir = root / "data"
    return _read_raw_parallel(data_dir / "en_sents", data_dir / "vi_sents")


def _get_shared_objects(config=None):
    if "objects" in _CACHE:
        return _CACHE["objects"]

    src_texts, trg_texts = _build_data(config, split="train")

    tokenizer_src = EnglishBPETokenizer(num_merges=int(_cfg(config, "num_merges", 50)))
    tokenizer_src.train(src_texts)

    vocab_src = Vocabulary(freq_threshold=int(_cfg(config, "src_freq_threshold", 1)))
    vocab_trg = Vocabulary(freq_threshold=int(_cfg(config, "trg_freq_threshold", 1)))

    src_tokens = [
        ["<sos>"] + tokenizer_src.encode(text) + ["<eos>"]
        for text in src_texts
    ]

    trg_tokens = [
        ["<sos>"] + tokenize_vi(text) + ["<eos>"]
        for text in trg_texts
    ]

    vocab_src.build_vocabulary(src_tokens)
    vocab_trg.build_vocabulary(trg_tokens)

    _CACHE["objects"] = tokenizer_src, vocab_src, vocab_trg
    return _CACHE["objects"]


class TargetVocabTokenizer:
    def __init__(self, vocab_trg):
        self.vocab_trg = vocab_trg
        self.pad_token_id = vocab_trg.stoi["<pad>"]

    def batch_decode(self, sequences, skip_special_tokens=True):
        special = {"<pad>", "<sos>", "<eos>", "<unk>"}
        outputs = []

        # Convert tensor to list if needed
        if torch.is_tensor(sequences):
            sequences = sequences.detach().cpu().tolist()

        for seq in sequences:
            tokens = []

            for idx in seq:
                # Handle tensor elements
                if torch.is_tensor(idx):
                    idx = int(idx.item())
                else:
                    idx = int(idx)
                    
                token = self.vocab_trg.itos.get(idx, "<unk>")

                if skip_special_tokens and token in special:
                    if token == "<eos>":
                        break
                    continue

                tokens.append(token.replace("_", " "))

            outputs.append(" ".join(tokens).strip())

        return outputs


def build_tokenizer(config=None):
    _, _, vocab_trg = _get_shared_objects(config)
    return TargetVocabTokenizer(vocab_trg)


def build_train_dataloader(config=None):
    src_texts, trg_texts = _build_data(config, split="train")
    tokenizer_src, vocab_src, vocab_trg = _get_shared_objects(config)

    batch_size = int(_cfg(config, "batch_size", 32))
    max_len = _cfg(config, "max_len", 0.95)

    return get_dataloader(
        src_texts=src_texts,
        trg_texts=trg_texts,
        vocab_src=vocab_src,
        vocab_trg=vocab_trg,
        tokenizer_src=tokenizer_src.encode,
        tokenizer_trg=tokenize_vi,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=True,
    )


def build_eval_dataloader(config=None):
    src_texts, trg_texts = _build_data(config, split="eval")
    tokenizer_src, vocab_src, vocab_trg = _get_shared_objects(config)

    eval_max_samples = _cfg(config, "eval_max_samples", 2000)
    if eval_max_samples is not None:
        eval_max_samples = min(int(eval_max_samples), len(src_texts))
        src_texts = src_texts[:eval_max_samples]
        trg_texts = trg_texts[:eval_max_samples]

    batch_size = int(_cfg(config, "eval_batch_size", _cfg(config, "batch_size", 32)))
    max_len = _cfg(config, "max_len", 0.95)

    return get_dataloader(
        src_texts=src_texts,
        trg_texts=trg_texts,
        vocab_src=vocab_src,
        vocab_trg=vocab_trg,
        tokenizer_src=tokenizer_src.encode,
        tokenizer_trg=tokenize_vi,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
    )


def build_model(config=None):
    _, vocab_src, vocab_trg = _get_shared_objects(config)

    hidden_size = int(_cfg(config, "hidden_size", 64))
    embed_dim = int(_cfg(config, "embed_dim", 64))
    attention_dim = int(_cfg(config, "attention_dim", 32))
    num_layers = int(_cfg(config, "num_layers", 1))
    cell_type = str(_cfg(config, "cell_type", "gru"))
    dropout = float(_cfg(config, "dropout", 0.1))

    encoder = Encoder(
        vocab_size=len(vocab_src),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type,
        dropout=dropout,
    )

    attention = BahdanauAttention(
        encoder_hidden_dim=hidden_size,
        decoder_hidden_dim=hidden_size,
        attention_dim=attention_dim,
    )

    decoder = Decoder(
        vocab_size=len(vocab_trg),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type,
        attention=attention,
        dropout=dropout,
        eos_token_id=vocab_trg.stoi["<eos>"],
    )

    return Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=vocab_src.stoi["<pad>"],
    )