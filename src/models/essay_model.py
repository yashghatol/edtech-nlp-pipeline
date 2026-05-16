"""DeBERTa regression head for holistic essay scoring."""

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class EssayScorer(nn.Module):
    """DeBERTa backbone + single regression head.

    Predicts a raw float; clip to [1, 6] and round at inference for QWK.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        """
        Args:
            model_name: HuggingFace model name or local path.
            dropout: Applied to [CLS] representation before regression head.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """Forward pass.

        Returns:
            (loss, logits): loss is None if labels not provided.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_repr = self.dropout(cls_repr)
        logits = self.regressor(cls_repr).squeeze(-1)   # shape: (batch_size,)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)

        return loss, logits

    def save_scorer(self, save_dir: str, tokenizer=None):
        """Save backbone + regressor head to directory.

        Args:
            save_dir: Path to output directory.
            tokenizer: If provided, save tokenizer too.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        torch.save(
            self.regressor.state_dict(),
            os.path.join(save_dir, "regressor_head.pt"),
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
if __name__ == "__main__":
    model = EssayScorer("microsoft/deberta-v3-base", dropout=0.1)
    print("Model loaded successfully")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Regressor: {model.regressor}")