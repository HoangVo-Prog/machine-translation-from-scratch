"""
Evaluate BLEU on a JSONL test file.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer_cache_dir checkpoints/tokenizers \
        --test_file data/valid_iwslt2015-en-vi.jsonl \
        --direction vi2en \
        --num_beams 5
"""

import argparse
import json
import torch
import sacrebleu
from tqdm import tqdm

from src.data.tokenizer import TranslationTokenizer
from src.builders.build_translator import Translator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_cache_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--direction", required=True)
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

    src_lang, tgt_lang = args.direction.split("2")
    src_sents, ref_sents = [], []
    with open(args.test_file) as f:
        for line in f:
            obj = json.loads(line)
            src_sents.append(obj[src_lang])
            ref_sents.append(obj[tgt_lang])

    hyp_sents = translator.translate(src_sents)
    bleu = sacrebleu.corpus_bleu(hyp_sents, [ref_sents])
    print(f"BLEU: {bleu.score:.2f}")
    print(bleu)


if __name__ == "__main__":
    main()
