import sys
import os

# Add project root to Python path so src/ is importable
sys.path.insert(0, os.path.abspath("."))

from src.utils.config import load_config, get_label_maps

cfg = load_config("configs/config.yaml")
label2id, id2label = get_label_maps(cfg)

print("Stage 1 model:", cfg["stage1_pii"]["model"]["pretrained_model"])
print("Stage 2 model:", cfg["stage2_essay"]["model"]["pretrained_model"])
print("Label O maps to ID:", label2id["O"])
print("ID 0 maps to label:", id2label[0])
print(f"Total BIO labels: {len(label2id)}")

# All 5 lines must print without error
assert cfg["stage1_pii"]["model"]["pretrained_model"] == "distilbert-base-uncased"
assert cfg["stage2_essay"]["model"]["pretrained_model"] == "microsoft/deberta-v3-small"
assert label2id["O"] == 0
assert id2label[0] == "O"
assert len(label2id) == 15

print("\nAll assertions passed. Phase 0 complete.")