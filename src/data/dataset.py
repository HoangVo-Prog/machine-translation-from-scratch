import random
import torch

# Lớp TranslationDataset: Kết nối dữ liệu thô với các bộ Tokenizer và Vocabulary
class TranslationDataset:
    def __init__(self, src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, max_len=0.95):
        self.max_len = max_len
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg

    def __getitem__(self, idx):
        # Tokenize source and target
        if hasattr(self.tokenizer_src, "encode"):
            src_tokens = self.tokenizer_src.encode(self.src_texts[idx])
        else:
            src_tokens = self.tokenizer_src(self.src_texts[idx])

        if hasattr(self.tokenizer_trg, "encode"):
            trg_tokens = self.tokenizer_trg.encode(self.trg_texts[idx])
        else:
            trg_tokens = self.tokenizer_trg(self.trg_texts[idx])

        src_limit = self._token_limit(src_tokens)
        trg_limit = self._token_limit(trg_tokens)

        src_tokens = src_tokens[:src_limit]
        trg_tokens = trg_tokens[:trg_limit]

        # Add <sos> and <eos>
        src_tokens = [self.vocab_src.stoi["<sos>"]] + src_tokens + [self.vocab_src.stoi["<eos>"]]
        trg_tokens = [self.vocab_trg.stoi["<sos>"]] + trg_tokens + [self.vocab_trg.stoi["<eos>"]]

        # Padding to max_len
        src_tokens = self._pad_sequence(src_tokens, self.vocab_src.stoi["<pad>"], src_limit)
        trg_tokens = self._pad_sequence(trg_tokens, self.vocab_trg.stoi["<pad>"], trg_limit)

        return torch.tensor(src_tokens), torch.tensor(trg_tokens)

    def __len__(self):
        return len(self.src_texts)

    def _pad_sequence(self, tokens, pad_idx, token_limit):
        pad_target = int(token_limit)
        while len(tokens) < pad_target + 2:  # +2 for <sos> and <eos>
            tokens.append(pad_idx)
        return tokens

    def _token_limit(self, tokens):
        if isinstance(self.max_len, float) and 0 < self.max_len < 1:
            return max(1, int(len(tokens) * self.max_len))
        return int(self.max_len)

# Lớp CollateBatch: Xử lý gộp các câu có độ dài khác nhau vào cùng một Batch
class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Tách riêng list câu Anh và Việt từ batch
        en_batch = [torch.tensor(item[0]) for item in batch]
        vi_batch = [torch.tensor(item[1]) for item in batch]

        # Padding động cho batch hiện tại
        en_padded = self._pad_batch(en_batch)
        vi_padded = self._pad_batch(vi_batch)

        # Tạo padding mask (True tại vị trí số 0/padding)
        en_mask = (en_padded == self.pad_idx)
        vi_mask = (vi_padded == self.pad_idx)

        return en_padded, vi_padded, en_mask, vi_mask

    def _pad_batch(self, batch):
        # Find the max length in the batch and pad all sequences to that length
        max_len = max([len(seq) for seq in batch])
        return torch.stack([torch.cat([seq, torch.tensor([self.pad_idx] * (max_len - len(seq)))]) for seq in batch])

# Hàm khởi tạo DataLoader để cung cấp dữ liệu cho quá trình huấn luyện
def get_dataloader(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=32, max_len=50, shuffle=False):
    dataset = TranslationDataset(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, max_len=max_len)

    collate_fn = CollateBatch(pad_idx=vocab_src.stoi['<pad>'])
    return TranslationDataLoader(dataset, batch_size, collate_fn, shuffle=shuffle)

class TranslationDataLoader:
    def __init__(self, dataset, batch_size, collate_fn, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_data = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch_data)


def custom_dataloader(dataset, batch_size, collate_fn):
    data_len = len(dataset)
    indices = list(range(data_len))
    
    for start_idx in range(0, data_len, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_data = [dataset[i] for i in batch_indices]
        
        # Process batch using collate_fn
        yield collate_fn(batch_data)