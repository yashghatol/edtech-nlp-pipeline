"""PyTorch Dataset for essay scoring."""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class EssayDataset(Dataset):
    """Tokenises essays and returns input tensors for DeBERTa."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        has_labels: bool = True,
    ):
        """
        Args:
            df: DataFrame with 'full_text' and optionally 'score'.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Truncate/pad to this many tokens.
            has_labels: False for test set (no 'score' column).
        """
        self.texts = df["full_text"].values
        self.labels = df["score"].values.astype(float) if has_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        # DeBERTa-v3 does not use token_type_ids — omit safely
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
'''if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer

    # Load a tiny slice of the data
    df = pd.read_csv("data/raw/essay/train.csv").head(4)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    dataset = EssayDataset(df, tokenizer, max_length=512)

    # Manually call __getitem__ on the first essay
    item = dataset[0]

    print("Keys in item:", item.keys())
    print("input_ids shape:", item["input_ids"].shape)
    print("attention_mask shape:", item["attention_mask"].shape)
    print("label:", item["labels"])'''