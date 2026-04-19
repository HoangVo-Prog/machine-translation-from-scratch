import re

def clean_text_en(text):
    text = text.lower().strip()
    # Thêm dấu nháy đơn (') vào để giữ lại các từ như don't, it's
    text = re.sub(r"([.!?'])", r" \1", text)
    text = re.sub(r"[^a-zA-Z0-9.!?']+", r" ", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text_vi(text):
    text = text.lower().strip()
    # Tiếng Việt không có từ viết tắt dùng dấu nháy đơn, chỉ giữ lại dấu câu cơ bản
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!?]+", r" ", text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# hàm tách từ
def tokenize_from_scratch(cleaned_text):
    """
    Dùng chung cho cả tiếng Anh và tiếng Việt (Syllable-level).
    """
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

def tokenize_en(text):
    cleaned = clean_text_en(text)
    return tokenize_from_scratch(cleaned)

def tokenize_vi(text):
    cleaned = clean_text_vi(text)
    return tokenize_from_scratch(cleaned)
