"""
Sanity-check a single sentence interactively.
"""

import argparse
import torch

from src.data.tokenizer import TranslationTokenizer
from src.builders.build_translator import Translator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_cache_dir", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--num_beams", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TranslationTokenizer.from_cache(args.tokenizer_cache_dir, args.direction)
    translator = Translator(
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.num_beams,
    )

    result = translator.translate([args.sentence])[0]
    print(f"Source     : {args.sentence}")
    print(f"Translation: {result}")


if __name__ == "__main__":
    main()
