class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                # Tự đếm tần suất
                frequencies[word] = frequencies.get(word, 0) + 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text_tokens):
        # Chuyển đổi danh sách từ (tokens) thành danh sách các chỉ số (indexes).
        return [self.stoi[token] if token in self.stoi else self.stoi["<unk>"] for token in text_tokens]
