import re
import string

class EnglishBPETokenizer:
    def __init__(self, num_merges=100, tokenizer_type="bpe"):
        self.num_merges = num_merges
        self.tokenizer_type = tokenizer_type
        self.merges = {}
        # Lưu sẵn all_marks để dùng trong clean_text
        self.all_marks = re.escape(string.punctuation)
        self.contractions = {
            "won't": "will not", "can't": "cannot", "i'm": "i am",
            "isn't": "is not", "aren't": "are not", "'m": " am",
            "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "n't": " not"
        }

    def clean_text_en(self, text):
        if not text:
            return ""

        # 1. Chuyển về chữ thường
        text = text.lower().strip()

        # 2. Xử lý viết tắt (Contractions)
        words = text.split()
        expanded_words = [self.contractions.get(w, w) for w in words]
        text = " ".join(expanded_words)

        # 3. Tách toàn bộ dấu câu
        text = re.sub(f"([{self.all_marks}])", r" \1 ", text)

        # 4. Xử lý số (Chuyển thành <num>)
        text = re.sub(r"\d+", " <num> ", text)

        # 5. Loại bỏ các ký tự lạ, giữ lại chữ, số, dấu câu và <>
        valid_chars = f"a-z0-9 " + self.all_marks + r"<> "
        text = re.sub(f"[^{valid_chars}]+", " ", text)

        # 6. Xử lý khoảng trắng thừa
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    # --- Các hàm bổ trợ BPE ---

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

    def _to_symbol_sequence(self, word):
        return [char for char in word] + ["</w>"]

    def _apply_bpe(self, sequence):
        # Tối ưu logic apply dựa trên merges đã học
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
        cleaned_corpus = [self.clean_text_en(text) for text in corpus]
        
        sequences = []
        for text in cleaned_corpus:
            for word in text.split():
                sequences.append(self._to_symbol_sequence(word))

        for i in range(self.num_merges):
            stats = self._get_stats(sequences)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_symbol = top_pair[0] + top_pair[1]
            sequences = self._merge(sequences, top_pair, new_symbol)
            self.merges[top_pair] = new_symbol

        return self.merges

    def encode(self, text, max_len=None):
        cleaned = self.clean_text_en(text)
        if self.tokenizer_type == "whitespace":
            tokens = cleaned.split()
        else:
            tokens = []
            for word in cleaned.split():
                sequence = self._to_symbol_sequence(word)
                if self.tokenizer_type == "bpe":
                    sequence = self._apply_bpe(sequence)
                token = "".join(sequence).replace("</w>", "")
                tokens.append(token)

        return tokens[:max_len] if max_len else tokens