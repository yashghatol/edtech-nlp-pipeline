# src/pipeline/essay_inference.py
"""End-to-end essay scoring: redacted text -> 1-6 score + LOO sentence importance."""

import spacy
import torch
import numpy as np
from transformers import AutoTokenizer

from src.models.essay_model import EssayScorer

# Load SpaCy once at module level for sentence segmentation.
# Each module that needs SpaCy loads it independently — sharing across
# modules is not straightforward and the 50ms load cost is acceptable.
nlp = spacy.load("en_core_web_sm")


def load_essay_model(fold_dir: str, device: torch.device):
    """Load EssayScorer backbone + regressor head from a fold directory.

    Two-step loading: EssayScorer saves backbone via save_pretrained() and
    regressor head separately as regressor_head.pt.
    Passing fold_dir as model_name makes AutoConfig and AutoModel load from disk.
    """
    model = EssayScorer(fold_dir).to(device).eval()
    model.regressor.load_state_dict(
        torch.load(
            f"{fold_dir}/regressor_head.pt",
            map_location=device,
            weights_only=True    # suppress FutureWarning in PyTorch >= 2.0
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(fold_dir, use_fast=True)
    return model, tokenizer


def score_essay(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 512
) -> float:
    """Score one essay string. Returns raw float prediction (not rounded).

    CRITICAL: EssayScorer.forward() returns (loss, logits) — a tuple.
    Unpack with '_, logits' because labels=None means loss=None always.
    Do NOT call model(...).item() directly — that calls .item() on a tuple.

    To convert to a display score: int(np.clip(np.round(raw), 1, 6))
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        # forward() signature: (input_ids, attention_mask, token_type_ids=None, labels=None)
        # Returns: (loss, logits) — loss is None when labels is None
        _, logits = model(
            enc['input_ids'].to(device),
            enc['attention_mask'].to(device)
        )
    return float(logits.item())


def compute_loo_importance(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 512
) -> tuple:
    """Compute sentence-level leave-one-out importance scores.

    For each sentence i, removes it and re-scores the remaining essay.
    importance_i = base_score - score_without_sentence_i

    Positive importance -> sentence raises the predicted score.
    Negative importance -> sentence lowers the predicted score.
    Zero -> single-sentence essay (undefined), or no change.

    Returns:
        (base_score: float, importances: list of dicts)
        Each dict: {'sentence': str, 'importance': float}

    Distribution shift note: This function receives PII-redacted text at
    inference but the model was trained on raw essays. [REDACTED] tokens are
    handled gracefully by DeBERTa's subword tokeniser. Scores are approximate.
    This is a documented known limitation — do not attempt to fix here.
    """
    doc       = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return 0.0, []

    base_score = score_essay(text, model, tokenizer, device, max_length)
    results    = []

    for i, sent in enumerate(sentences):
        remaining = " ".join(s for j, s in enumerate(sentences) if j != i)

        if not remaining.strip():
            # Edge case: single-sentence essay — LOO is undefined
            results.append({'sentence': sent, 'importance': 0.0})
            continue

        loo_score  = score_essay(remaining, model, tokenizer, device, max_length)
        importance = base_score - loo_score
        results.append({
            'sentence':   sent,
            'importance': round(importance, 4)
        })

    return base_score, results