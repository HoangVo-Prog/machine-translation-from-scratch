import json
import re
import os
import unicodedata
import string

class VietnameseTokenizer:
    def __init__(self, tokenizer_type="bpe", num_merges=100, dict_path="src/data/vi_words.txt"):
        self.tokenizer_type = tokenizer_type
        self.num_merges = num_merges
        self.merges = {}
        self.compound_words = self._load_vi_words(dict_path)
        self.all_marks = re.escape(string.punctuation)
        self.cache = {}

    # --- Các hàm bổ trợ (Tiền xử lý) ---

    def save(self, path: str):
        data = {
            "tokenizer_type": self.tokenizer_type,
            "num_merges": self.num_merges,
            "merges": {f"{p[0]}|{p[1]}": v for p, v in self.merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, dict_path: str = "src/data/vi_words.txt") -> "VietnameseTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(
            num_merges=data["num_merges"],
            tokenizer_type=data["tokenizer_type"],
            dict_path=dict_path,
        )
        tok.merges = {tuple(k.split("|", 1)): v for k, v in data["merges"].items()}
        return tok

    def _load_vi_words(self, file_path):
        if not os.path.exists(file_path):
            return set()
        with open(file_path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    def clean_text_vi(self, text):
        if not text:
            return ""
        # 1. Chuẩn hóa Unicode & chữ thường
        text = unicodedata.normalize('NFC', text).lower()
        
        # 2. Tách dấu câu
        text = re.sub(f"([{self.all_marks}])", r" \1 ", text)
        
        # 3. Lọc ký tự lạ
        valid_chars = f"a-z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ " + self.all_marks
        text = re.sub(f"[^{valid_chars}]+", " ", text)
        
        # 4. Xử lý khoảng trắng
        return re.sub(r"\s+", " ", text).strip()

    # --- Logic xử lý Token ---

    def _to_symbol_sequence(self, token):
        # Thêm </w> để đánh dấu kết thúc một đơn vị từ/từ ghép
        return [char for char in token] + ["</w>"]

    def _get_compound_tokens(self, cleaned_text):
        """Tách từ ghép trước khi chạy BPE"""
        syllables = cleaned_text.split()
        tokens = []
        i = 0
        n = len(syllables)

        while i < n:
            matched = False
            for length in range(4, 1, -1): # Ưu tiên từ dài (4 âm tiết) xuống 2
                if i + length <= n:
                    phrase = " ".join(syllables[i:i + length])
                    if phrase in self.compound_words:
                        tokens.append(phrase.replace(" ", "_"))
                        i += length
                        matched = True
                        break
            if not matched:
                tokens.append(syllables[i])
                i += 1
        return tokens

    # --- Logic BPE (Train & Apply) ---

    def _get_stats(self, sequences):
        counts = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, sequences, pair, new_symbol):
        merged = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                    new_seq.append(new_symbol)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            merged.append(new_seq)
        return merged

    def _apply_bpe(self, sequence):
        if not self.merges: # Nếu chưa train, thoát luôn
            return sequence
        
        for pair, new_symbol in self.merges.items():
            i = 0
            while i < len(sequence) - 1:
                if (sequence[i], sequence[i + 1]) == pair:
                    sequence = sequence[:i] + [new_symbol] + sequence[i + 2:]
                else:
                    i += 1
        return sequence

    # --- Giao diện chính ---

    def train(self, corpus):
        """Huấn luyện bộ Merges trên tập dữ liệu lớn"""
        cleaned_corpus = [self.clean_text_vi(text) for text in corpus]
        
        # Bước trung gian: Tách từ ghép để BPE học trên các unit lớn
        initial_tokens = []
        for line in cleaned_corpus:
            initial_tokens.extend(self._get_compound_tokens(line))

        sequences = [self._to_symbol_sequence(t) for t in initial_tokens]

        for i in range(self.num_merges):
            stats = self._get_stats(sequences)
            if not stats: break
            top_pair = max(stats, key=stats.get)
            new_symbol = top_pair[0] + top_pair[1]
            sequences = self._merge(sequences, top_pair, new_symbol)
            self.merges[top_pair] = new_symbol
        
        return self.merges

    def encode(self, text, max_len=None):
        """Chuyển text thành danh sách token"""
        cleaned = self.clean_text_vi(text)
        
        if self.tokenizer_type == "whitespace":
            return cleaned.split()[:max_len] if max_len else cleaned.split()
            
        # 1. Tách từ ghép/âm tiết
        compound_tokens = self._get_compound_tokens(cleaned)
        all_tokens = []
        
        # 2. Chạy BPE trên từng từ
        for word in compound_tokens:
            # Kiểm tra cache
            if word in self.cache:
                all_tokens.extend(self.cache[word])
                continue
            
            # Nếu chưa có trong cache thì tính toán
            sequence = self._to_symbol_sequence(word)
            sequence = self._apply_bpe(sequence)
            token_list = [s.replace("</w>", "") for s in sequence if s != "</w>"]
            
            # Lưu vào cache và thêm vào kết quả
            self.cache[word] = token_list
            all_tokens.extend(token_list)

        return all_tokens[:max_len] if max_len else all_tokens