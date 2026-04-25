import os
import json
import random

# ====== CONFIG ======
SRC_FILE = "data/en_sents"
TRG_FILE = "data/vi_sents"
OUTPUT_DIR = "data/processed"
TRAIN_RATIO = 0.9
SEED = 42

# ====== LOAD DATA ======
with open(SRC_FILE, "r", encoding="utf-8") as f:
    src_lines = [line.strip() for line in f if line.strip()]

with open(TRG_FILE, "r", encoding="utf-8") as f:
    trg_lines = [line.strip() for line in f if line.strip()]

assert len(src_lines) == len(trg_lines), "Số dòng không khớp!"

pairs = list(zip(src_lines, trg_lines))

# ====== SHUFFLE ======
random.seed(SEED)
random.shuffle(pairs)

# ====== SPLIT ======
split_idx = int(len(pairs) * TRAIN_RATIO)
train_pairs = pairs[:split_idx]
valid_pairs = pairs[split_idx:]

print(f"Total: {len(pairs)}")
print(f"Train: {len(train_pairs)}")
print(f"Valid: {len(valid_pairs)}")

# ====== SAVE ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_jsonl(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for src, trg in pairs:
            obj = {"src": src, "trg": trg}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

save_jsonl(train_pairs, os.path.join(OUTPUT_DIR, "train.jsonl"))
save_jsonl(valid_pairs, os.path.join(OUTPUT_DIR, "valid.jsonl"))

print("Done! Files saved to:", OUTPUT_DIR)