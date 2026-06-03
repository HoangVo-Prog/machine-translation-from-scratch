"""
Translate sentences from a file using a saved checkpoint.

Usage:
    python scripts/translate.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer_cache_dir checkpoints/tokenizers \
        --direction vi2en \
        --input_file input.txt \
        --output_file output.txt \
        --num_beams 5
"""

import argparse
import torch
from pathlib import Path

from src.data.tokenizer import TranslationTokenizer
from src.builders.build_translator import Translator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_cache_dir", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="translations.txt")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=150)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TranslationTokenizer.from_cache(args.tokenizer_cache_dir, args.direction)

    translator = Translator(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.num_beams,
        max_len=args.max_len,
    )

    sentences = Path(args.input_file).read_text().strip().splitlines()
    translations = translator.translate(sentences)

    with open(args.output_file, "w") as f:
        for t in translations:
            f.write(t + "\n")
    print(f"Saved {len(translations)} translations to {args.output_file}")


if __name__ == "__main__":
    main()
