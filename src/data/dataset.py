import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi):
        self.en_sentences = en_sentences
        self.vi_sentences = vi_sentences
        self.vocab_en = vocab_en
        self.vocab_vi = vocab_vi
        self.tokenize_en = tokenize_en
        self.tokenize_vi = tokenize_vi

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, index):
        en_text = self.en_sentences[index]
        vi_text = self.vi_sentences[index]

        # Tokenize và thêm <sos>, <eos>
        en_tokens = ["<sos>"] + self.tokenize_en(en_text) + ["<eos>"]
        vi_tokens = ["<sos>"] + self.tokenize_vi(vi_text) + ["<eos>"]

        return (
            torch.tensor(self.vocab_en.numericalize(en_tokens)),
            torch.tensor(self.vocab_vi.numericalize(vi_tokens))
        )

class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        en_batch = [item[0] for item in batch]
        vi_batch = [item[1] for item in batch]

        # Tự động thêm padding để các sequence trong batch bằng độ dài nhau
        en_batch = pad_sequence(en_batch, batch_first=True, padding_value=self.pad_idx)
        vi_batch = pad_sequence(vi_batch, batch_first=True, padding_value=self.pad_idx)

        return en_batch, vi_batch

def get_dataloader(en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi, batch_size=32, shuffle=True):
    dataset = TranslationDataset(en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi)
    pad_idx = vocab_en.stoi["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CollateBatch(pad_idx)
    )
    return loader
