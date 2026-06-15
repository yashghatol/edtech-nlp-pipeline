# src/models/pii_model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils.hub_utils import resolve_model_file

class PIITokenClassifier(nn.Module):
    """DeBERTa-v3-base backbone + per-token linear head for 13-class BIO NER."""

    def __init__(self, model_name_or_path: str, num_labels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.backbone   = AutoModel.from_pretrained(model_name_or_path)
        hidden_size     = self.backbone.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_head()

    def _init_head(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def enable_gradient_checkpointing(self):
        """Reduce VRAM by recomputing activations during backward.

        ⚠️ use_reentrant=False is mandatory for PyTorch 2.x.
        This was the Phase 2 training crash. Do not remove the kwarg.
        """
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> dict:
        """Returns {'logits': Tensor(B, seq_len, num_labels)}."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,   # None for DeBERTa; BERT needs it
        )
        x      = self.dropout(outputs.last_hidden_state)  # (B, L, H)
        logits = self.classifier(x)                        # (B, L, 13)
        return {"logits": logits}

    @classmethod
    def from_saved(cls, model_dir: str, num_labels: int,
                   dropout: float = 0.1) -> "PIITokenClassifier":
        """Load backbone from directory + classifier head from .pt file.

        Used in Phase 3 (Streamlit). Mirrors Phase 2 two-step load pattern.
        """
        model = cls(model_dir, num_labels, dropout)
        model.classifier.load_state_dict(
            torch.load(resolve_model_file(model_dir, "classifier_head.pt"), map_location=...)
        )
        return model