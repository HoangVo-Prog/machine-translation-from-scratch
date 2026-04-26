import re
import os


def clean_text_vi(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!? ]+", r" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_vi_words(file_path="src/vi_words.txt"):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if " " in line.strip()}

VI_COMPOUND_WORDS = load_vi_words()

class VietnameseTokenizer:
    def __init__(self, tokenizer_type="bpe", num_merges=100):
        self.tokenizer_type = tokenizer_type
        self.num_merges = num_merges
        self.merges = {}

    def _to_symbol_sequence(self, syllable):
        return [char for char in syllable] + ["</w>"]

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
        while True:
            found = False
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                if pair in self.merges:
                    sequence = sequence[:i] + [self.merges[pair]] + sequence[i + 2:]
                    found = True
                    break
            if not found:
                break
        return sequence

    def _split_syllables(self, cleaned):
        syllables = []
        word = ""
        for char in cleaned:
            if char == " ":
                if word:
                    syllables.append(word)
                word = ""
            else:
                word += char
        if word:
            syllables.append(word)
        return syllables

    def _compound_tokens(self, cleaned):
        syllables = self._split_syllables(cleaned)
        tokens = []
        i = 0
        n = len(syllables)

        while i < n:
            matched = False
            for length in range(4, 1, -1):
                if i + length <= n:
                    phrase = " ".join(syllables[i:i + length])
                    if phrase in VI_COMPOUND_WORDS:
                        tokens.append(phrase.replace(" ", "_"))
                        i += length
                        matched = True
                        break
            if not matched:
                tokens.append(syllables[i])
                i += 1
        return tokens

    def encode(self, text, max_len=None):
        cleaned = clean_text_vi(text)
        if self.tokenizer_type == "whitespace":
            tokens = cleaned.split()
        elif self.tokenizer_type == "compound":
            tokens = self._compound_tokens(cleaned)
        else:
            tokens = []
            for syllable in cleaned.split():
                sequence = self._to_symbol_sequence(syllable)
                if self.tokenizer_type == "bpe":
                    sequence = self._apply_bpe(sequence)
                token = "".join(sequence).replace("</w>", "")
                tokens.append(token)

        if max_len is not None:
            return tokens[:max_len]
        return tokens

    def tokenize(self, text, max_len=None):
        return self.encode(text, max_len=max_len)

    def train(self, corpus):
        cleaned_corpus = [clean_text_vi(text) for text in corpus]
        if self.tokenizer_type in {"whitespace", "compound"}:
            return [line.split() for line in cleaned_corpus]

        sequences = []
        for line in cleaned_corpus:
            for syllable in line.split():
                sequences.append(self._to_symbol_sequence(syllable))

        for i in range(self.num_merges):
            stats = self._get_stats(sequences)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_symbol = top_pair[0] + top_pair[1]
            sequences = self._merge(sequences, top_pair, new_symbol)
            self.merges[top_pair] = new_symbol

        return sequences


def tokenize_vi(text, max_len=None, tokenizer_type="bpe"):
    tokenizer = VietnameseTokenizer(tokenizer_type=tokenizer_type)
    return tokenizer.encode(text, max_len=max_len)
