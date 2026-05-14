import json
import logging
from pathlib import Path
import torch

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.models.attentions import BahdanauAttention

from src.data.dataset import TranslationDataLoader, CollateBatch, TranslationDataset, get_dataloader
from src.data.vocab import Vocabulary
from src.data.vi_tokenizer import VietnameseTokenizer
from src.data.en_tokenizer import EnglishBPETokenizer


_CACHE = {}
LOGGER = logging.getLogger(__name__)


def _cfg(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _normalize_max_len(raw_max_len, default: int = 128, min_len: int = 4) -> int:
    try:
        value = int(raw_max_len)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid max_len=%r. Falling back to default=%s.", raw_max_len, default)
        value = default

    if value < min_len:
        LOGGER.warning("max_len=%s is too small. Clamping to min_len=%s.", value, min_len)
        value = min_len
    return value


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

    root = _repo_root()
    cache_dir = root / _cfg(config, "tokenizer_cache_dir", "checkpoints/tokenizers")

    src_ckpt      = cache_dir / "tokenizer_src.json"
    trg_ckpt      = cache_dir / "tokenizer_trg.json"
    vocab_src_ckpt = cache_dir / "vocab_src.pkl"
    vocab_trg_ckpt = cache_dir / "vocab_trg.pkl"
    ids_ckpt       = cache_dir / "train_ids.pt"

    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    src_texts, trg_texts = _build_data(config, split="train")
    max_len = _normalize_max_len(_cfg(config, "max_len", 128))

    # --- Load tokenizer ---
    if src_ckpt.exists() and trg_ckpt.exists():
        LOGGER.info("Tìm thấy BPE checkpoint, bỏ qua train BPE.")
        tokenizer_src = EnglishBPETokenizer.load(str(src_ckpt))
        tokenizer_trg = VietnameseTokenizer.load(str(trg_ckpt))
    else:
        LOGGER.info("Không có BPE checkpoint, bắt đầu train BPE...")
        tokenizer_src = EnglishBPETokenizer(tokenizer_type="bpe", num_merges=int(_cfg(config, "num_merges", 50)))
        tokenizer_src.train(src_texts)
        tokenizer_src.save(str(src_ckpt))

        tokenizer_trg = VietnameseTokenizer(tokenizer_type="bpe", num_merges=int(_cfg(config, "num_merges", 50)))
        tokenizer_trg.train(trg_texts)
        tokenizer_trg.save(str(trg_ckpt))
        LOGGER.info("Đã lưu BPE checkpoint vào %s", cache_dir)

    # --- Load vocab + IDs nếu đã có checkpoint ---
    if vocab_src_ckpt.exists() and vocab_trg_ckpt.exists() and ids_ckpt.exists():
        LOGGER.info("Tìm thấy vocab + IDs checkpoint, bỏ qua tokenizing.")
        import pickle
        with open(vocab_src_ckpt, "rb") as f:
            vocab_src = pickle.load(f)
        with open(vocab_trg_ckpt, "rb") as f:
            vocab_trg = pickle.load(f)
        saved = torch.load(ids_ckpt)
        src_ids = saved["src_ids"]
        trg_ids = saved["trg_ids"]
    else:
        # --- Tokenize + build vocab + numericalize ---
        vocab_src = Vocabulary(freq_threshold=int(_cfg(config, "src_freq_threshold", 1)))
        vocab_trg = Vocabulary(freq_threshold=int(_cfg(config, "trg_freq_threshold", 1)))

        print(f"Tokenizing {len(src_texts)} câu (1 lần duy nhất)...")
        src_tokens = [["<sos>"] + tokenizer_src.encode(text)[:max_len - 2] + ["<eos>"] for text in src_texts]
        trg_tokens = [["<sos>"] + tokenizer_trg.encode(text)[:max_len - 2] + ["<eos>"] for text in trg_texts]

        vocab_src.build_vocabulary(src_tokens)
        vocab_trg.build_vocabulary(trg_tokens)

        print("Numericalizing...")
        src_ids = [torch.tensor([vocab_src.stoi.get(t, vocab_src.stoi["<unk>"]) for t in toks], dtype=torch.long)
                   for toks in src_tokens]
        trg_ids = [torch.tensor([vocab_trg.stoi.get(t, vocab_trg.stoi["<unk>"]) for t in toks], dtype=torch.long)
                   for toks in trg_tokens]
        print("Hoàn tất tiền xử lý!")

        # --- Lưu vocab + IDs vào cache_dir ---
        # Chỉ lưu nếu cache_dir writable
        try:
            import pickle
            with open(vocab_src_ckpt, "wb") as f:
                pickle.dump(vocab_src, f)
            with open(vocab_trg_ckpt, "wb") as f:
                pickle.dump(vocab_trg, f)
            torch.save({"src_ids": src_ids, "trg_ids": trg_ids}, ids_ckpt)
            LOGGER.info("Đã lưu vocab + IDs vào %s", cache_dir)
        except OSError as e:
            LOGGER.warning("Không lưu được vocab/IDs cache: %s", e)

    _CACHE["objects"] = tokenizer_src, tokenizer_trg, vocab_src, vocab_trg
    _CACHE["train_data"] = src_texts, trg_texts
    _CACHE["train_ids"] = src_ids, trg_ids
    return _CACHE["objects"]


def build_tokenizer(config):
    tokenizer_src, tokenizer_trg, _, _ = _get_shared_objects(config)
    return tokenizer_src, tokenizer_trg


def build_train_dataloader(config=None):
    tokenizer_src, tokenizer_trg, vocab_src, vocab_trg = _get_shared_objects(config)
    src_texts, trg_texts = _CACHE["train_data"]
    src_ids, trg_ids = _CACHE["train_ids"]          # ★ lấy IDs đã xử lý

    batch_size = int(_cfg(config, "batch_size", 32))
    max_len = _normalize_max_len(_cfg(config, "max_len", 128))

    dataset = TranslationDataset(
        src_texts, trg_texts, vocab_src, vocab_trg,
        tokenizer_src, tokenizer_trg, max_len=max_len,
        src_ids=src_ids, trg_ids=trg_ids,           # ★ bypass tokenization
    )
    collate_fn = CollateBatch(pad_idx_src=vocab_src.stoi['<pad>'],
                              pad_idx_trg=vocab_trg.stoi['<pad>'])
    return TranslationDataLoader(dataset, batch_size, collate_fn, shuffle=True)


def build_eval_dataloader(config=None):
    src_texts, trg_texts = _build_data(config, split="eval")
    tokenizer_src, tokenizer_trg, vocab_src, vocab_trg = _get_shared_objects(config)

    eval_max_samples = _cfg(config, "eval_max_samples", 2000)
    if eval_max_samples is not None:
        eval_max_samples = min(int(eval_max_samples), len(src_texts))
        src_texts = src_texts[:eval_max_samples]
        trg_texts = trg_texts[:eval_max_samples]

    batch_size = int(_cfg(config, "eval_batch_size", _cfg(config, "batch_size", 32)))
    max_len = _normalize_max_len(_cfg(config, "max_len", 128))

    return get_dataloader(
        src_texts=src_texts,
        trg_texts=trg_texts,
        vocab_src=vocab_src,
        vocab_trg=vocab_trg,
        tokenizer_src=tokenizer_src,
        tokenizer_trg=tokenizer_trg,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
    )


def build_model(config=None):
    tokenizer_src, tokenizer_trg, vocab_src, vocab_trg = _get_shared_objects(config)

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