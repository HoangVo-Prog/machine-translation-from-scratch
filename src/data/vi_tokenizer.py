import re

class VietnameseBPETokenizer:
    def __init__(self, num_merges=1000):
        # Số lượng quy tắc gộp từ (càng cao thì càng nhiều từ ghép được tạo ra)
        self.num_merges = num_merges
        # Danh sách các quy tắc gộp: {(cặp_âm_tiết): từ_mới}
        self.merges = {}

    def clean_text_vi(self, text):
        # Chuyển về chữ thường
        text = text.lower().strip()
        # Tách dấu câu bằng regex cơ bản (được phép dùng re để xử lý chuỗi nhanh)
        text = re.sub(r"([.!?])", r" \1 ", text)
        # Giữ lại chữ cái Tiếng Việt, số và khoảng trắng
        text = re.sub(r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9.!? ]+", r" ", text)

        # Tự viết code xóa khoảng trắng thừa thay vì dùng .split() rồi .join()
        res = ""
        last_char = ""
        for char in text:
            if char == " " and last_char == " ":
                continue
            res += char
            last_char = char
        return res.strip()

    def train(self, corpus):
        """
        Mô hình TỰ HỌC: Không dùng file txt, tự đếm tần suất để tìm từ ghép.
        """
        # Khởi tạo: Chia câu thành các âm tiết rời rạc
        # Ví dụ: "học sinh học bài" -> [['học', 'sinh', 'học', 'bài']]
        splits = []
        for text in corpus:
            cleaned = self.clean_text_vi(text)
            syllables = []
            word = ""
            for char in cleaned:
                if char == " ":
                    if word: syllables.append(word)
                    word = ""
                else:
                    word += char
            if word: syllables.append(word)
            splits.append(syllables)

        # Bắt đầu vòng lặp học các cặp từ đi chung với nhau
        for _ in range(self.num_merges):
            # 1. Tự code đếm tần suất cặp âm tiết (Không dùng Counter)
            pair_counts = {}
            for syllables in splits:
                for j in range(len(syllables) - 1):
                    pair = (syllables[j], syllables[j+1])
                    if pair in pair_counts:
                        pair_counts[pair] += 1
                    else:
                        pair_counts[pair] = 1

            if not pair_counts:
                break

            # 2. Tự code tìm cặp xuất hiện nhiều nhất (Không dùng hàm max())
            best_pair = None
            max_freq = 0
            for pair in pair_counts:
                if pair_counts[pair] > max_freq:
                    max_freq = pair_counts[pair]
                    best_pair = pair

            # Nếu không còn cặp nào xuất hiện trên 1 lần thì dừng
            if max_freq < 2:
                break

            # 3. Ghi nhớ quy tắc gộp: (sinh, viên) -> sinh_viên
            new_token = best_pair[0] + "_" + best_pair[1]
            self.merges[best_pair] = new_token

            # 4. Cập nhật lại toàn bộ dữ liệu với từ mới vừa gộp
            new_splits = []
            for syllables in splits:
                new_s = []
                j = 0
                while j < len(syllables):
                    if j < len(syllables) - 1 and (syllables[j], syllables[j+1]) == best_pair:
                        new_s.append(new_token)
                        j += 2
                    else:
                        new_s.append(syllables[j])
                        j += 1
                new_splits.append(new_s)
            splits = new_splits

    def encode(self, text):
        """
        Áp dụng quy tắc đã học để tách từ cho một câu mới
        """
        cleaned = self.clean_text_vi(text)
        # Tách thành âm tiết
        syllables = []
        word = ""
        for char in cleaned:
            if char == " ":
                if word: syllables.append(word)
                word = ""
            else: word += char
        if word: syllables.append(word)

        # Lặp để gộp từ theo những gì đã học (merges)
        while len(syllables) >= 2:
            best_pair_to_merge = None
            # Trong BPE, ta phải gộp theo đúng thứ tự các quy tắc đã học
            for pair in self.merges:
                found = False
                for i in range(len(syllables) - 1):
                    if (syllables[i], syllables[i+1]) == pair:
                        best_pair_to_merge = pair
                        found = True
                        break
                if found: break

            if best_pair_to_merge is None:
                break

            # Thực hiện gộp
            new_token = self.merges[best_pair_to_merge]
            new_s = []
            i = 0
            while i < len(syllables):
                if i < len(syllables) - 1 and (syllables[i], syllables[i+1]) == best_pair_to_merge:
                    new_s.append(new_token)
                    i += 2
                else:
                    new_s.append(syllables[i])
                    i += 1
            syllables = new_s

        return syllables
