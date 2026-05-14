"""
translate.py — Inference EN → VI từ checkpoint đã train.

Yêu cầu trong thư mục checkpoint:
    model.pt
    vocab_src.pkl
    vocab_trg.pkl
    tokenizer_src.json
    tokenizer_trg.json

Cách dùng:
    # Interactive
    python translate.py --checkpoint checkpoints/best

    # Dịch 1 câu thẳng
    python translate.py --checkpoint checkpoints/best --text "I love you"

    # Chỉ định kiến trúc nếu khác default
    python translate.py --checkpoint checkpoints/best --hidden_size 256 --embed_dim 256
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.models.attentions import BahdanauAttention
from src.data.en_tokenizer import EnglishBPETokenizer
from src.data.vi_tokenizer import VietnameseTokenizer


# ── Load ────────────────────────────────────────────────────────────────────────

def load_all(checkpoint_dir: str):
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_dir}")

    # Vocab
    with open(ckpt / "vocab_src.pkl", "rb") as f:
        vocab_src = pickle.load(f)
    with open(ckpt / "vocab_trg.pkl", "rb") as f:
        vocab_trg = pickle.load(f)

    # Tokenizer
    tokenizer_src = EnglishBPETokenizer.load(str(ckpt / "tokenizer_src.json"))
    tokenizer_trg = VietnameseTokenizer.load(str(ckpt / "tokenizer_trg.json"))

    return vocab_src, vocab_trg, tokenizer_src, tokenizer_trg


def build_model(vocab_src, vocab_trg, args, device) -> Seq2Seq:
    encoder = Encoder(
        vocab_size=len(vocab_src),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        dropout=0.0,  # inference không cần dropout
    )

    attention = BahdanauAttention(
        encoder_hidden_dim=args.hidden_size,
        decoder_hidden_dim=args.hidden_size,
        attention_dim=args.attention_dim,
    )

    decoder = Decoder(
        vocab_size=len(vocab_trg),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        attention=attention,
        dropout=0.0,
        eos_token_id=vocab_trg.stoi["<eos>"],
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=vocab_src.stoi["<pad>"],
    )

    state_dict = torch.load(
        Path(args.checkpoint) / "model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── Translate ───────────────────────────────────────────────────────────────────

def translate(
    text: str,
    model: Seq2Seq,
    tokenizer_src,
    vocab_src,
    vocab_trg,
    device: torch.device,
    max_len: int = 100,
) -> str:
    # 1. Encode câu nguồn
    tokens = tokenizer_src.encode(text)
    ids = (
        [vocab_src.stoi["<sos>"]]
        + [vocab_src.stoi.get(t, vocab_src.stoi["<unk>"]) for t in tokens]
        + [vocab_src.stoi["<eos>"]]
    )
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, src_len]

    # 2. Greedy decode
    with torch.no_grad():
        pred_ids = model.greedy_decode(
            src=src,
            bos_token_id=vocab_trg.stoi["<sos>"],
            max_length=max_len,
        )  # [1, tgt_len]

    # 3. IDs → text, bỏ special tokens
    special = {vocab_trg.stoi["<sos>"], vocab_trg.stoi["<eos>"], vocab_trg.stoi["<pad>"]}
    pred_tokens = [
        vocab_trg.itos[idx.item()]
        for idx in pred_ids[0]
        if idx.item() not in special
    ]
    return " ".join(pred_tokens)


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dịch EN → VI")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Thư mục checkpoint, vd: checkpoints/best")

    # Inference
    parser.add_argument("--text", type=str, default=None,
                        help="Câu EN cần dịch (bỏ trống để chạy interactive)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Độ dài tối đa câu dịch")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (mặc định tự detect)")

    # Kiến trúc model — phải khớp với lúc train
    parser.add_argument("--hidden_size",   type=int, default=64)
    parser.add_argument("--embed_dim",     type=int, default=64)
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--num_layers",    type=int, default=1)
    parser.add_argument("--cell_type",     type=str, default="gru")

    args = parser.parse_args()

    # Device
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}") 

    # Load
    print(f"Đang load checkpoint: {args.checkpoint}")
    vocab_src, vocab_trg, tokenizer_src, tokenizer_trg = load_all(args.checkpoint)
    print(f"Vocab src: {len(vocab_src)} | Vocab trg: {len(vocab_trg)}")

    model = build_model(vocab_src, vocab_trg, args, device)
    print("Load xong!\n")

    def _run(text: str) -> str:
        return translate(text, model, tokenizer_src, vocab_src, vocab_trg, device, args.max_len)

    # Dịch 1 câu
    if args.text:
        print(f"EN: {args.text}")
        print(f"VI: {_run(args.text)}")
        return

    # Interactive
    print("Nhập câu tiếng Anh để dịch (gõ 'q' để thoát):\n")
    while True:
        try:
            text = input("EN: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            print("Thoát.")
            break

        print(f"VI: {_run(text)}\n")


if __name__ == "__main__":
    main()
