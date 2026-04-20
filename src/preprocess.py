import re
import os

def clean_text_en(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9.!?']+", r" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_en(text):
    # Tự code tách từ không dùng .split()
    cleaned = clean_text_en(text)
    tokens = []
    word = ""
    for char in cleaned:
        if char == " ":
            if word:
                tokens.append(word)
                word = ""
        else:
            word += char
    if word:
        tokens.append(word)
    return tokens

def clean_text_vi(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!?]+", r" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Load từ điển từ file bên ngoài
def load_vi_words(file_path="src/vi_words.txt"):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        words = {line.strip().lower() for line in f if " " in line.strip()}
    return words

VI_COMPOUND_WORDS = load_vi_words()

def tokenize_vi(text):
    cleaned = clean_text_vi(text)
    # Tách âm tiết thủ công
    syllables = []
    word = ""
    for char in cleaned:
        if char == " ":
            if word: syllables.append(word)
            word = ""
        else: word += char
    if word: syllables.append(word)
    
    tokens = []
    i = 0
    n = len(syllables)

    while i < n:
        matched = False
        for length in range(4, 1, -1):
            if i + length <= n:
                phrase = " ".join(syllables[i:i+length])
                if phrase in VI_COMPOUND_WORDS:
                    tokens.append(phrase.replace(" ", "_"))
                    i += length
                    matched = True
                    break
        if not matched:
            tokens.append(syllables[i])
            i += 1
    return tokens
