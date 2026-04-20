from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = {}  # Lưu trữ các cặp đã gộp: (char1, char2) -> merged_token
        self.vocab = []

    def get_stats(self, ids):
        """Thống kê tần suất các cặp token cạnh nhau"""
        counts = {}
        for word_id_list in ids:
            for i in range(len(word_id_list) - 1):
                pair = (word_id_list[i], word_id_list[i+1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """Gộp cặp token xuất hiện nhiều nhất thành token mới"""
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
        # Bước 1: Khởi tạo với từng ký tự đơn lẻ (UTF-8 encoding)
        # ids là danh sách các list, mỗi list chứa mã code của các ký tự trong từ
        ids = [list(word.encode("utf-8")) for word in corpus]
        
        vocab_size = 256 # Gốc là 256 byte đầu tiên
        
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            if not stats: break
            
            # Bước 2: Tìm cặp xuất hiện nhiều nhất
            top_pair = max(stats, key=stats.get)
            
            # Bước 3: Gộp cặp đó lại thành ID mới
            new_id = vocab_size + i
            ids = self.merge(ids, top_pair, new_id)
            
            # Lưu lại quy tắc merge
            self.merges[top_pair] = new_id
            print(f"Merge {i+1}: {top_pair} -> {new_id} (count: {stats[top_pair]})")
            
        return ids
