import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Lớp TranslationDataset: Kết nối dữ liệu thô với các bộ Tokenizer và Vocabulary
class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg):
        self.data = []

        for src, trg in zip(src_texts, trg_texts):
            # 1. Tách từ và chèn <sos>, <eos> dạng chữ
            en_tokens = ["<sos>"] + tokenizer_src(src) + ["<eos>"]
            vi_tokens = ["<sos>"] + tokenizer_trg(trg) + ["<eos>"]

            # 2. Dùng chính hàm numericalize xịn xò của ông để chuyển chữ thành số
            src_encoded = vocab_src.numericalize(en_tokens)
            trg_encoded = vocab_trg.numericalize(vi_tokens)

            self.data.append((src_encoded, trg_encoded))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Lớp CollateBatch: Xử lý gộp các câu có độ dài khác nhau vào cùng một Batch
class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Tách riêng list câu Anh và Việt từ batch
        en_batch = [torch.tensor(item[0]) for item in batch]
        vi_batch = [torch.tensor(item[1]) for item in batch]

        # Padding động cho batch hiện tại
        en_padded = pad_sequence(en_batch, batch_first=True, padding_value=self.pad_idx)
        vi_padded = pad_sequence(vi_batch, batch_first=True, padding_value=self.pad_idx)

        # Tạo padding mask (True tại vị trí số 0/padding)
        en_mask = (en_padded == self.pad_idx)
        vi_mask = (vi_padded == self.pad_idx)

        return en_padded, vi_padded, en_mask, vi_mask

# Hàm khởi tạo DataLoader để cung cấp dữ liệu cho quá trình huấn luyện
def get_dataloader(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=32):
    dataset = TranslationDataset(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg)

    # Lấy pad_idx từ dictionary stoi trong class Vocabulary của ông
    collate_fn = CollateBatch(pad_idx=vocab_src.stoi['<pad>'])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return dataloader
