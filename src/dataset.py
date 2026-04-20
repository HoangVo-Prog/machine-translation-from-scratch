import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Lớp TranslationDataset kế thừa từ torch.utils.data.Dataset
class TranslationDataset(Dataset):
    def __init__(self, en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi):
        self.en_sentences = en_sentences
        self.vi_sentences = vi_sentences
        self.vocab_en = vocab_en
        self.vocab_vi = vocab_vi
        self.tokenize_en = tokenize_en
        self.tokenize_vi = tokenize_vi

    # Trả về số lượng cặp câu trong tập dữ liệu
    def __len__(self):
        return len(self.en_sentences)

    # Lấy một cặp câu theo chỉ mục
    def __getitem__(self, index):
        en_text = self.en_sentences[index]
        vi_text = self.vi_sentences[index]

        # Tokenize và bao bọc bằng token đặc biệt <sos> (start of sentence) và <eos> (end of sentence)
        en_tokens = ["<sos>"] + self.tokenize_en(en_text) + ["<eos>"]
        vi_tokens = ["<sos>"] + self.tokenize_vi(vi_text) + ["<eos>"]

        # Chuyển đổi danh sách token thành Tensor số bằng cách sử dụng bộ từ vựng đã xây dựng
        return (
            torch.tensor(self.vocab_en.numericalize(en_tokens)),
            torch.tensor(self.vocab_vi.numericalize(vi_tokens))
        )

# Lớp CollateBatch để xử lý việc gộp các mẫu đơn lẻ thành một batch
class CollateBatch:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Tách các câu tiếng Anh và tiếng Việt từ batch
        en_batch = [item[0] for item in batch]
        vi_batch = [item[1] for item in batch]

        # Tự động thêm padding để các sequence trong batch có cùng độ dài
        en_batch = pad_sequence(en_batch, batch_first=True, padding_value=self.pad_idx)
        vi_batch = pad_sequence(vi_batch, batch_first=True, padding_value=self.pad_idx)

        return en_batch, vi_batch

# Hàm trợ giúp để tạo DataLoader
def get_dataloader(en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi, batch_size=32, shuffle=True):
    # Khởi tạo TranslationDataset
    dataset = TranslationDataset(en_sentences, vi_sentences, vocab_en, vocab_vi, tokenize_en, tokenize_vi)
    # Lấy chỉ mục của token padding từ bộ từ vựng tiếng Anh
    pad_idx = vocab_en.stoi["<pad>"]

    # Khởi tạo DataLoader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CollateBatch(pad_idx) # Sử dụng CollateBatch để xử lý padding
    )
    return loader
