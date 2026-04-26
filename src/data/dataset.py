import random
import torch

class TranslationDataset:
    def __init__(self, src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, max_len=128):
        self.max_len = max_len
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg

    def __getitem__(self, idx):
        # 1. Tokenize
        src_tokens = self.tokenizer_src.encode(self.src_texts[idx]) if hasattr(self.tokenizer_src, "encode") else self.tokenizer_src(self.src_texts[idx])
        trg_tokens = self.tokenizer_trg.encode(self.trg_texts[idx]) if hasattr(self.tokenizer_trg, "encode") else self.tokenizer_trg(self.trg_texts[idx])

        # 2. Cắt câu nếu quá dài (Tránh nổ bộ nhớ)
        src_tokens = src_tokens[:self.max_len - 2] # -2 cho sos và eos
        trg_tokens = trg_tokens[:self.max_len - 2]

        # 3. Chuyển sang ID số
        src_ids = [self.vocab_src.stoi["<sos>"]] + self.vocab_src.numericalize(src_tokens) + [self.vocab_src.stoi["<eos>"]]
        trg_ids = [self.vocab_trg.stoi["<sos>"]] + self.vocab_trg.numericalize(trg_tokens) + [self.vocab_trg.stoi["<eos>"]]

        # KHÔNG PAD Ở ĐÂY - Trả về tensor độ dài thực thực tế
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

    def __len__(self):
        return len(self.src_texts)

class CollateBatch:
    def __init__(self, pad_idx_src, pad_idx_trg):
        self.pad_idx_src = pad_idx_src
        self.pad_idx_trg = pad_idx_trg

    def __call__(self, batch):
        src_batch, trg_batch = zip(*batch)

        # Dynamic Padding: Chỉ pad đến câu dài nhất TRONG BATCH NÀY
        src_padded = self._pad_sequences(src_batch, self.pad_idx_src)
        trg_padded = self._pad_sequences(trg_batch, self.pad_idx_trg)

        # Tạo mask: Vị trí nào là padding thì True (hoặc 1)
        # Lưu ý: Transformer thường cần mask kiểu (batch, 1, 1, seq_len) hoặc (batch, seq_len)
        src_mask = (src_padded == self.pad_idx_src)
        trg_mask = (trg_padded == self.pad_idx_trg)

        return src_padded, trg_padded, src_mask, trg_mask

    def _pad_sequences(self, sequences, pad_idx):
        max_len = max([len(seq) for seq in sequences])
        # Tạo tensor chứa toàn pad_idx trước
        padded_seqs = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
        # Copy dữ liệu thực vào
        for i, seq in enumerate(sequences):
            padded_seqs[i, :len(seq)] = seq
        return padded_seqs

def get_dataloader(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=32, max_len=128, shuffle=False):
    dataset = TranslationDataset(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, max_len=max_len)
    
    # Pass cả 2 pad_idx để an toàn
    collate_fn = CollateBatch(pad_idx_src=vocab_src.stoi['<pad>'], pad_idx_trg=vocab_trg.stoi['<pad>'])
    
    return TranslationDataLoader(dataset, batch_size, collate_fn, shuffle=shuffle)

class TranslationDataLoader:
    def __init__(self, dataset, batch_size, collate_fn, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.indices = list(range(len(dataset)))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_idxs = self.indices[i : i + self.batch_size]
            batch_samples = [self.dataset[idx] for idx in batch_idxs]
            yield self.collate_fn(batch_samples)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size