import re
import os
from collections import Counter

class EnglishBPETokenizer:
    def __init__(self, num_merges=100):
        # Số lượng phép hợp nhất (merges) tối đa
        self.num_merges = num_merges
        # Lưu trữ các quy tắc gộp (merges)
        self.merges = {}
        # Danh sách các từ viết tắt tiếng Anh phổ biến (contractions)
        self.contractions = {
            "won't": "will not", "can't": "cannot", "i'm": "i am",
            "isn't": "is not", "aren't": "are not", "'m": " am",
            "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "n't": " not"
        }

    def clean_text(self, text):
        # Chuyển về chữ thường và loại bỏ khoảng trắng ở hai đầu
        text = text.lower().strip()
        # Xử lý các từ viết tắt (contractions)
        for word in text.split():
            if word in self.contractions:
                text = text.replace(word, self.contractions[word])
        # Thêm khoảng trắng xung quanh các dấu câu
        text = re.sub(r"([.!?\"\(\),:;])", r" \1 ", text)
        # Thay thế các chữ số bằng token <num>
        text = re.sub(r"\d+", " <num> ", text)
        # Chỉ giữ lại các ký tự hợp lệ
        text = re.sub(r"[^a-zA-Z0-9.!?\"\(\),:;<num> ]+", r" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def get_stats(self, ids):
        # Thống kê tần suất xuất hiện của các cặp byte cạnh nhau
        counts = {}
        for word_id_list in ids:
            for i in range(len(word_id_list) - 1):
                pair = (word_id_list[i], word_id_list[i+1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        # Gộp cặp byte đã chọn thành một ID mới
        new_ids = []
        for word_id_list in ids:
            new_word = []
            i = 0
            while i < len(word_id_list):
                if i < len(word_id_list) - 1 and (word_id_list[i], word_id_list[i+1]) == pair:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word_id_list[i])
                    i += 1
            new_ids.append(new_word)
        return new_ids

    def train(self, corpus):
        # Huấn luyện tokenizer để tìm ra các quy tắc gộp BPE
        cleaned_corpus = [self.clean_text(text) for text in corpus]
        # Chuyển văn bản thành danh sách byte (UTF-8 encoding)
        ids = [list(text.encode("utf-8")) for text in cleaned_corpus]
        vocab_size = 256 # Giá trị khởi đầu cho byte đơn
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            # Chọn cặp byte xuất hiện nhiều nhất để gộp (merge)
            top_pair = max(stats, key=stats.get)
            new_id = vocab_size + i
            ids = self.merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
        return ids

    def encode(self, text):
        # Mã hóa một đoạn văn bản thành danh sách subword tokens (chuỗi số)
        text = self.clean_text(text)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = {}
            for i in range(len(tokens) - 1):
                stats[(tokens[i], tokens[i+1])] = i

            # Tìm cặp merge có độ ưu tiên cao nhất dựa trên bộ quy tắc BPE đã huấn luyện
            best_pair = None
            for pair in self.merges:
                if pair in stats:
                    best_pair = pair
                    break

            if best_pair is None: break

            new_id = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Trả về các token dưới dạng chuỗi để đưa vào Vocabulary
        return [str(t) for t in tokens]
