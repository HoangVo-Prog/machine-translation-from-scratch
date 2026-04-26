import torch
import random

def pad_custom(batch_list, pad_idx):
    max_len = max(len(seq) for seq in batch_list)
    padded_batch = []

    for seq in batch_list:
        # Số lượng pad cần thêm = max_len - chiều dài hiện tại
        pads_needed = max_len - len(seq)
        padded_seq = seq + [pad_idx] * pads_needed
        padded_batch.append(padded_seq)

    return torch.tensor(padded_batch, dtype=torch.long)


class CustomBatchIterator:
    def __init__(self, src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_idx_src = vocab_src.stoi['<pad>']
        self.pad_idx_trg = vocab_trg.stoi['<pad>']

        self.data = []
        # Chuyển đổi toàn bộ chữ thành số 1 lần khi khởi tạo
        for src, trg in zip(src_texts, trg_texts):
            en_tokens = ["<sos>"] + tokenizer_src(src) + ["<eos>"]
            vi_tokens = ["<sos>"] + tokenizer_trg(trg) + ["<eos>"]

            src_encoded = vocab_src.numericalize(en_tokens)
            trg_encoded = vocab_trg.numericalize(vi_tokens)

            self.data.append((src_encoded, trg_encoded))

        # Tính tổng số batch
        self.num_batches = (len(self.data) + batch_size - 1) // batch_size

    def __iter__(self):
        # 1. Xáo trộn dữ liệu nếu cần
        if self.shuffle:
            random.shuffle(self.data)

        # 2. Cắt dữ liệu thành từng mẻ (batch)
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]

            # Tách Anh - Việt
            en_batch = [item[0] for item in batch]
            vi_batch = [item[1] for item in batch]

            # 3. Padding thủ công
            en_padded = pad_custom(en_batch, self.pad_idx_src)
            vi_padded = pad_custom(vi_batch, self.pad_idx_trg)

            # 4. Sinh Mask thủ công
            en_mask = (en_padded == self.pad_idx_src)
            vi_mask = (vi_padded == self.pad_idx_trg)

            # 5. Dùng yield để trả về từng batch (Generator)
            yield en_padded, vi_padded, en_mask, vi_mask

    def __len__(self):
        return self.num_batches

# Giữ nguyên tên hàm này để file Train không phải sửa code import
def get_dataloader(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=32):
    return CustomBatchIterator(src_texts, trg_texts, vocab_src, vocab_trg, tokenizer_src, tokenizer_trg, batch_size=batch_size, shuffle=True)
