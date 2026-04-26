import os
import pandas as pd
import json
import random

# ====== CONFIG ======
SRC_FILE = "data/PhoMT.csv"
OUTPUT_DIR = "data/processed"
TRAIN_RATIO = 0.9
SEED = 42

# ====== LOAD DATA ======
df = pd.read_csv(SRC_FILE, header=None, names=["EnglishSentences", "VietnameseSentences"])

pairs = list(zip(df["EnglishSentences"], df["VietnameseSentences"]))

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

save_jsonl(train_pairs, os.path.join(OUTPUT_DIR, "train_phomt.jsonl"))
save_jsonl(valid_pairs, os.path.join(OUTPUT_DIR, "valid_phomt.jsonl"))

print("Done! Files saved to:", OUTPUT_DIR)