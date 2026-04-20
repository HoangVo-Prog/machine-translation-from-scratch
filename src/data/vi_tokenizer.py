import re
import os

def clean_text_vi(text):
    text = text.lower().strip()
    # Xử lý dấu câu
    text = re.sub(r"([.!?])", r" \1 ", text)
    # Giữ lại các ký tự tiếng Việt, số và dấu câu cơ bản
    text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!? ]+", r" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_vi_words(file_path="src/vi_words.txt"):
    # Tải danh sách từ ghép tiếng Việt từ file
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        # Chỉ lấy các dòng có khoảng trắng (từ ghép)
        words = {line.strip().lower() for line in f if " " in line.strip()}
    return words

# Khởi tạo từ điển
VI_COMPOUND_WORDS = load_vi_words()

def tokenize_vi(text):
    # Tách từ tiếng Việt sử dụng thuật toán Longest Matching
    cleaned = clean_text_vi(text)
    
    # Tách âm tiết thủ công (không dùng .split)
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
        # Tìm kiếm từ ghép dài nhất (tối đa 4 âm tiết)
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
