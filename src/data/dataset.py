import random
import torch

class TranslationDataset:
    def __init__(self, src_texts, trg_texts, vocab_src, vocab_trg,
                 tokenizer_src, tokenizer_trg, max_len=128,
                 src_ids=None, trg_ids=None):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.vocab_trg = vocab_trg         
        self.src_data = []
        self.trg_data = []

        # Nếu đã có IDs sẵn, bỏ qua tokenization hoàn toàn
        if src_ids is not None and trg_ids is not None:
            print(f"Đang load {len(src_ids)} câu từ cache (không tokenize lại)...")
            self.src_data = src_ids
            self.trg_data = trg_ids
            print("Hoàn tất!")
            return

        # tokenize bình thường (dùng cho eval set)
        print(f"Đang tiền xử lý {len(src_texts)} câu...")
        for s_text, t_text in zip(src_texts, trg_texts):
            s_tokens = tokenizer_src.encode(s_text)[:max_len - 2]
            t_tokens = tokenizer_trg.encode(t_text)[:max_len - 2]
            s_ids = [vocab_src.stoi["<sos>"]] + vocab_src.numericalize(s_tokens) + [vocab_src.stoi["<eos>"]]
            t_ids = [vocab_trg.stoi["<sos>"]] + vocab_trg.numericalize(t_tokens) + [vocab_trg.stoi["<eos>"]]
            self.src_data.append(torch.tensor(s_ids, dtype=torch.long))
            self.trg_data.append(torch.tensor(t_ids, dtype=torch.long))
        print("Tiền xử lý hoàn tất!")

    def __getitem__(self, idx):
        # Chỉ việc lấy data đã xử lý xong, cực nhanh
        return self.src_data[idx], self.trg_data[idx]

    def __len__(self):
        return len(self.src_data)

class CollateBatch:
    def __init__(self, pad_idx_src, pad_idx_trg):
        self.pad_idx_src = pad_idx_src
        self.pad_idx_trg = pad_idx_trg

    def __call__(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_padded = self._pad_sequences(src_batch, self.pad_idx_src)
        trg_padded = self._pad_sequences(trg_batch, self.pad_idx_trg)
        src_mask = (src_padded == self.pad_idx_src)
        trg_mask = (trg_padded == self.pad_idx_trg)

        # Tính độ dài thực của từng câu source
        src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)

        return src_padded, trg_padded, src_mask, trg_mask, src_lengths

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