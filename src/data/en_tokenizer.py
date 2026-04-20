import re
import os
from collections import Counter

class EnglishBPETokenizer:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
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

    def get_stats(self, ids):
        counts = {}
        for word_id_list in ids:
            for i in range(len(word_id_list) - 1):
                pair = (word_id_list[i], word_id_list[i+1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
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
        # Làm sạch và chuyển thành bytes
        cleaned_corpus = [self.clean_text(text) for text in corpus]
        ids = [list(text.encode("utf-8")) for text in cleaned_corpus]
        
        vocab_size = 256
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            top_pair = max(stats, key=stats.get)
            new_id = vocab_size + i
            ids = self.merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
        return ids
