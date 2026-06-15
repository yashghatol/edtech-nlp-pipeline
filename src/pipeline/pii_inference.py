# src/pipeline/pii_inference.py
"""End-to-end PII inference: raw text -> token-level predictions with char offsets."""

import spacy
import torch
import numpy as np
from transformers import AutoTokenizer

from src.models.pii_model import PIITokenClassifier
from src.evaluation.pii_metrics import apply_entity_threshold
from src.data.pii_dataset import LABEL2ID, ID2LABEL, PIIDataset

# Load SpaCy once at module import — not inside functions.
# This avoids reloading the 12 MB model on every call at inference time.
nlp = spacy.load("en_core_web_sm")


def load_pii_model(model_dir: str, cfg: dict, device: torch.device):
    """Load PIITokenClassifier and tokenizer from disk. Call once at startup."""
    model = PIITokenClassifier.from_saved(
        model_dir,
        num_labels=len(LABEL2ID),          # 13 — must match training
        dropout=cfg['stage1']['dropout']
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    return model, tokenizer


def run_pii_inference(
    text: str,
    model,
    tokenizer,
    cfg: dict,
    device: torch.device
) -> list:
    """
    Tokenize text with SpaCy, run PII model, return per-token predictions.

    Returns list of dicts:
        [{'token': str, 'label': str, 'start': int, 'end': int}, ...]

    'label' is 'O' or a BIO tag string (e.g. 'B-NAME_STUDENT').
    'start' and 'end' are character offsets in the original text string.
    These offsets are used by apply_redactions() to reconstruct the text.
    """
    # Step A: SpaCy tokenization — must match training tokenization.
    # Do NOT use text.split(). SpaCy handles punctuation attachment differently
    # (e.g. "nathalie@example.com." splits the trailing dot as a separate token).
    doc   = nlp(text)
    words = [token.text for token in doc]
    spans = [(token.idx, token.idx + len(token.text)) for token in doc]

    if not words:
        return []

    # Step B: Build a single-record PIIDataset item with dummy labels.
    # PIIDataset handles subword alignment (is_split_into_words=True) internally.
    # The dummy 'O' labels are discarded — we only need input_ids and attention_mask.
    record = {
        'document': 'inference',
        'tokens':   words,
        'labels':   ['O'] * len(words)    # dummy — not used during inference
    }
    ds   = PIIDataset(
        [record], tokenizer, LABEL2ID,
        max_length=cfg['stage1']['max_length']
    )
    item = ds[0]

    # Step C: Forward pass.
    # PIITokenClassifier returns a dict-like object with a 'logits' key.
    # Shape of logits: (1, seq_len, num_labels) -> after [0]: (seq_len, num_labels)
    with torch.no_grad():
        out = model(
            input_ids=item['input_ids'].unsqueeze(0).to(device),
            attention_mask=item['attention_mask'].unsqueeze(0).to(device)
        )
    logits = out['logits'][0].cpu().float().numpy()    # (seq_len, 13)

    # Step D: Extract first-subword logits only — one vector per word.
    # Re-tokenize here to get word_ids because PIIDataset does not expose
    # its internal word_ids mapping.
    word_ids_map = tokenizer(
        words,
        is_split_into_words=True,
        max_length=cfg['stage1']['max_length'],
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).word_ids(batch_index=0)

    word_logits, prev_wid = [], None
    for pos, wid in enumerate(word_ids_map):
        if wid is not None and wid != prev_wid:
            word_logits.append(logits[pos])
        prev_wid = wid

    word_logits = np.array(word_logits)    # (num_words, 13)

    # Step E: Apply per-entity confidence floors instead of raw argmax.
    # URL_PERSONAL and USERNAME over-fire badly with raw argmax (Phase 1 learning).
    pred_labels = apply_entity_threshold(
        word_logits,
        ID2LABEL,
        entity_floors=cfg['stage1']['entity_floors']
    )

    # Step F: Zip words, labels, and character offsets into output dicts.
    results = []
    for word, label, (start, end) in zip(words, pred_labels, spans):
        results.append({
            'token': word,
            'label': label,
            'start': start,
            'end':   end
        })
    return results


def apply_redactions(text: str, predictions: list, redacted_indices: set) -> str:
    """
    Replace user-selected tokens with [REDACTED] in the original text.

    Args:
        text:              Original raw essay string.
        predictions:       Output of run_pii_inference() — list of token dicts.
        redacted_indices:  Set of integer positions (indices into predictions list)
                           chosen by the user to redact.

    Returns:
        Redacted text string with selected tokens replaced by [REDACTED].

    Note: Replacements are applied in reverse offset order so that replacing
    a later character range does not shift the start/end positions of earlier
    ranges that have not yet been processed.
    """
    result = text
    for i in sorted(redacted_indices, reverse=True):
        p      = predictions[i]
        result = result[:p['start']] + '[REDACTED]' + result[p['end']:]
    return result