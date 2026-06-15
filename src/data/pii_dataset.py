# src/data/pii_dataset.py

import json
import random
import torch
from torch.utils.data import Dataset

# 🔴 CORRECTED: 13 labels based on T0 data inspection.
# I-EMAIL and I-USERNAME do not exist in the competition training data —
# those entity types are always single SpaCy tokens, never multi-token spans.
PII_LABELS = [
    "O",
    "B-NAME_STUDENT", "I-NAME_STUDENT",   # 1365 + 1096 examples
    "B-EMAIL",                              # 39 examples  (no I-EMAIL in data)
    "B-USERNAME",                           # 6 examples   (no I-USERNAME in data)
    "B-ID_NUM",        "I-ID_NUM",          # 78 + 1 examples
    "B-PHONE_NUM",     "I-PHONE_NUM",       # 6 + 15 examples
    "B-URL_PERSONAL",  "I-URL_PERSONAL",    # 110 + 1 examples
    "B-STREET_ADDRESS","I-STREET_ADDRESS",  # 2 + 20 examples
]
# len(PII_LABELS) == 13

LABEL2ID = {label: i for i, label in enumerate(PII_LABELS)}
ID2LABEL  = {i: label for i, label in enumerate(PII_LABELS)}


def load_pii_records(json_path: str) -> list:
    """Load train.json → list of dicts with 'document', 'tokens', 'labels'."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "document": doc["document"],
            "tokens":   doc["tokens"],
            "labels":   doc["labels"],
        }
        for doc in data
    ]


def train_val_split(records: list, val_fraction: float = 0.2,
                    random_state: int = 42) -> tuple:
    """Split at DOCUMENT level to prevent essay-level leakage.

    ⚠️ LEAKAGE: Never split at token or sentence level.
    One essay must be entirely in train OR entirely in val.
    """
    rng     = random.Random(random_state)
    doc_ids = list({r["document"] for r in records})
    rng.shuffle(doc_ids)
    n_val   = int(len(doc_ids) * val_fraction)
    val_ids = set(doc_ids[:n_val])
    return (
        [r for r in records if r["document"] not in val_ids],
        [r for r in records if r["document"] in val_ids],
    )


class PIIDataset(Dataset):
    """PyTorch Dataset: word-level BIO labels aligned to DeBERTa subword tokens.

    Alignment rule:
    - First subword of each word  → word's BIO label
    - Continuation subwords        → -100  (ignored by loss and seqeval)
    - Special tokens ([CLS], [SEP], [PAD]) → -100
    """

    def __init__(self, records: list, tokenizer, label2id: dict,
                 max_length: int = 512):
        self.records    = records
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record      = self.records[idx]
        words       = record["tokens"]
        word_labels = record["labels"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids     = encoding.word_ids(batch_index=0)
        label_ids    = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(self.label2id[word_labels[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }