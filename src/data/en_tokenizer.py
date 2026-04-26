import re

class EnglishBPETokenizer:
    def __init__(self, num_merges=100, tokenizer_type="bpe"):
        self.num_merges = num_merges
        self.tokenizer_type = tokenizer_type
        self.merges = {}
        self.contractions = {
            "won't": "will not", "can't": "cannot", "i'm": "i am",
            "isn't": "is not", "aren't": "are not", "'m": " am",
            "'re": " are", "'s": " is", "'ll": " will",
            "'ve": " have", "'d": " would", "n't": " not"
        }

    def clean_text(self, text):
        text = text.lower().strip()
        for word in text.split():
            if word in self.contractions:
                text = text.replace(word, self.contractions[word])
        text = re.sub(r"([.!?\"\(\),:;])", r" \1 ", text)
        text = re.sub(r"\d+", " <num> ", text)
        text = re.sub(r"[^a-zA-Z0-9.!?\"\(\),:;<num> ]+", r" ", text)
        return re.sub(r"\s+", " ", text).strip()

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

    def train(self, corpus):
        cleaned_corpus = [self.clean_text(text) for text in corpus]
        if self.tokenizer_type == "whitespace":
            return [text.split() for text in cleaned_corpus]

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

        return sequences

    def encode(self, text, max_len=None):
        cleaned = self.clean_text(text)
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

        if max_len is not None:
            return tokens[:max_len]
        return tokens

    def tokenize(self, text, max_len=None):
        return self.encode(text, max_len=max_len)


def build_english_tokenizer(tokenizer_type="bpe", num_merges=100):
    return EnglishBPETokenizer(num_merges=num_merges, tokenizer_type=tokenizer_type)
