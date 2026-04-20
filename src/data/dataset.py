import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Lớp TranslationDataset: Kết nối dữ liệu thô với các bộ Tokenizer và Vocabulary
class TranslationDataset(Dataset):
    def __init__(self, en_sentences, vi_sentences, vocab_en, vocab_vi, en_tokenizer_obj, tokenize_vi_func):
        self.en_sentences = en_sentences
        self.vi_sentences = vi_sentences
        self.vocab_en = vocab_en
        self.vocab_vi = vocab_vi
        self.en_tokenizer_obj = en_tokenizer_obj # Đối tượng EnglishBPETokenizer đã train
        self.tokenize_vi_func = tokenize_vi_func # Hàm tokenize_vi (Longest Matching)

    def __len__(self):
        # Trả về tổng số lượng mẫu trong tập dữ liệu
        return len(self.en_sentences)

    def __getitem__(self, index):
        # Lấy một cặp câu Anh-Việt theo chỉ mục
        en_text = self.en_sentences[index]
        vi_text = self.vi_sentences[index]

        # Tiền xử lý tiếng Anh: Sử dụng phương thức encode của BPE (không dùng .split())
        en_tokens = ["<sos>"] + self.en_tokenizer_obj.encode(en_text) + ["<eos>"]

        # Tiền xử lý tiếng Việt: Dùng thuật toán Longest Matching
        vi_tokens = ["<sos>"] + self.tokenize_vi_func(vi_text) + ["<eos>"]

        # Chuyển đổi tokens thành dạng số (Tensor) dựa trên bộ từ điển tương ứng
        return (
            torch.tensor(self.vocab_en.numericalize(en_tokens)),
            torch.tensor(self.vocab_vi.numericalize(vi_tokens))
        )

# Lớp CollateBatch: Xử lý gộp các câu có độ dài khác nhau vào cùng một Batch
class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx # Chỉ số của token <pad> để lấp đầy các khoảng trống

    def __call__(self, batch):
        # Tách danh sách cặp tensor (Anh, Việt)
        en_batch = [item[0] for item in batch]
        vi_batch = [item[1] for item in batch]

        # Thêm padding để tất cả các câu trong batch có cùng độ dài với câu dài nhất
        en_batch = pad_sequence(en_batch, batch_first=True, padding_value=self.pad_idx)
        vi_batch = pad_sequence(vi_batch, batch_first=True, padding_value=self.pad_idx)

        return en_batch, vi_batch

# Hàm khởi tạo DataLoader để cung cấp dữ liệu cho quá trình huấn luyện
def get_dataloader(en_sentences, vi_sentences, vocab_en, vocab_vi, en_tokenizer_obj, tokenize_vi_func, batch_size=32, shuffle=True):
    dataset = TranslationDataset(en_sentences, vi_sentences, vocab_en, vocab_vi, en_tokenizer_obj, tokenize_vi_func)
    pad_idx = vocab_en.stoi["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CollateBatch(pad_idx) # Sử dụng hàm collate để xử lý padding động
    )
    return loader
