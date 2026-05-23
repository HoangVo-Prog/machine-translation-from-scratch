"""
High-level translator: load a saved checkpoint and translate sentences.
"""

import torch
from pathlib import Path
from typing import List

from src.models.seq2seq import Seq2Seq
from src.data.tokenizer import TranslationTokenizer


class Translator:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer: TranslationTokenizer,
        device: torch.device = None,
        num_beams: int = 5,
        max_len: int = 150,
    ):
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.num_beams = num_beams
        self.max_len = max_len

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        args_dict = ckpt["args"]

        self.model = Seq2Seq(
            src_vocab_size=tokenizer.src.vocab_size,
            tgt_vocab_size=tokenizer.tgt.vocab_size,
            embed_dim=args_dict.get("embed_dim", 256),
            hidden_size=args_dict.get("hidden_size", 512),
            num_layers=args_dict["num_layers"],
            cell_type=args_dict["cell_type"],
            attention_type=args_dict["attention_type"],
            bidirectional=args_dict["bidirectional"],
            dropout=0.0,
            src_pad_idx=tokenizer.src.pad_idx,
            tgt_pad_idx=tokenizer.tgt.pad_idx,
            sos_idx=tokenizer.tgt.bos_idx,
            eos_idx=tokenizer.tgt.eos_idx,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, sentences: List[str]) -> List[str]:
        results = []
        for sent in sentences:
            ids = self.tokenizer.src.encode(sent, add_bos=False, add_eos=True)
            src = torch.tensor([ids], dtype=torch.long, device=self.device)
            src_lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

            best_seqs = self.model.beam_search(
                src, src_lengths, num_beams=self.num_beams, max_len=self.max_len
            )
            results.append(self.tokenizer.tgt.decode(best_seqs[0]))
        return results
