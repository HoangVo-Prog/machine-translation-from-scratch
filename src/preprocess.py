import re
from underthesea import word_tokenize

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!?]+", r" ", text)
    return text.strip()

def tokenize_en(text):
    cleaned_text = clean_text(text)
    words = []
    current_word = ""

    # Duyệt qua từng ký tự để tự tách từ theo khoảng trắng
    for char in cleaned_text:
        if char != ' ':
            current_word += char
        else:
            if current_word:
                words.append(current_word)
                current_word = ""

    # Thêm từ cuối cùng nếu có
    if current_word:
        words.append(current_word)

    return words

def tokenize_vi(text):
    # Tiếng Việt vẫn dùng thư viện để đảm bảo độ chính xác cho mô hình
    return word_tokenize(clean_text(text))
