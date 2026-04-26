from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=2):
        # Giữ nguyên các token đặc biệt
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        # 1. Đếm toàn bộ tần suất trước
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1

        # 2. Lọc và gán ID
        idx = 4 # Bắt đầu sau các token đặc biệt
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text_tokens):
        # Chuyển token thành ID, dùng <unk> nếu không có trong từ điển
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in text_tokens]