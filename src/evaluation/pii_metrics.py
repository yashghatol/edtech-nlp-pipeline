# src/evaluation/pii_metrics.py

import numpy as np
import torch
from seqeval.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.metrics import f1_score as sklearn_f1


def compute_entity_f1(true_seqs: list, pred_seqs: list) -> dict:
    """Entity-level F1 via seqeval — the competition metric.

    An entity is correct only if ALL its tokens have correct labels.
    Partial match (right type, wrong boundary) = wrong.

    Args:
        true_seqs: List[List[str]] — one list per document, word-level BIO labels
        pred_seqs: List[List[str]] — same structure, predicted labels
    """
    return {
        "entity_f1":        f1_score(true_seqs, pred_seqs),
        "entity_precision": precision_score(true_seqs, pred_seqs),
        "entity_recall":    recall_score(true_seqs, pred_seqs),
        "report":           classification_report(true_seqs, pred_seqs),
    }


def compute_token_f1(true_seqs: list, pred_seqs: list, label2id: dict) -> float:
    """Token-level macro F1 — debugging only, NOT the competition metric.

    If entity F1 is low but token F1 is decent, boundaries are off (B→I).
    If both are low, individual token predictions are wrong (model issue).
    """
    flat_true = [label2id.get(t, 0) for seq in true_seqs for t in seq]
    flat_pred = [label2id.get(p, 0) for seq in pred_seqs for p in seq]
    return sklearn_f1(flat_true, flat_pred, average="macro")


def apply_threshold(logits: np.ndarray, id2label: dict,
                    threshold: float = 0.5) -> list:
    """Convert logits to label strings with an adjustable O-class threshold.

    threshold = 0.5: standard argmax.
    threshold = 0.9: predict O only if P(O) > 0.9; otherwise predict best PII.

    Args:
        logits:    np.ndarray shape (seq_len, num_labels) for ONE sequence.
        id2label:  dict mapping int IDs → label strings.
        threshold: float in [0, 1].

    Returns:
        List[str] of label strings, length = seq_len.
    """
    probs     = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    o_probs   = probs[:, 0]
    pii_probs = probs[:, 1:]
    best_pii  = pii_probs.argmax(axis=-1) + 1   # +1: offset into full label space
    preds     = probs.argmax(axis=-1)

    mask        = o_probs < threshold
    preds[mask] = best_pii[mask]

    return [id2label[int(p)] for p in preds]


def threshold_sweep(all_logits: list, all_true: list, id2label: dict,
                    thresholds: list) -> dict:
    """Sweep O-class thresholds on the val set and return metrics for each.

    ⚠️ LEAKAGE: Run only on val set. Never on test data.

    Args:
        all_logits: List[np.ndarray] — one per document, shape (num_words, num_labels)
        all_true:   List[List[str]] — word-level true label sequences
        id2label:   dict mapping int IDs → label strings
        thresholds: List[float]

    Returns:
        dict: threshold → {'f1': float, 'precision': float, 'recall': float}
    """
    results = {}
    for t in thresholds:
        all_pred = [apply_threshold(logits, id2label, threshold=t)
                    for logits in all_logits]
        metrics  = compute_entity_f1(all_true, all_pred)
        results[t] = {
            "f1":        metrics["entity_f1"],
            "precision": metrics["entity_precision"],
            "recall":    metrics["entity_recall"],
        }
    return results
DEFAULT_ENTITY_FLOORS = {
    "B-URL_PERSONAL": 0.80,   # Tightened: no external data, rely on high confidence only
    "I-URL_PERSONAL": 0.80,
    "B-ID_NUM":       0.55,
    "I-ID_NUM":       0.55,
    "B-NAME_STUDENT": 0.40,
    "I-NAME_STUDENT": 0.40,
}

def apply_entity_threshold(logits: np.ndarray, id2label: dict,
                            o_threshold: float = 0.5,
                            entity_floors: dict = None) -> list:
    if entity_floors is None:
        entity_floors = DEFAULT_ENTITY_FLOORS
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    ).numpy()
    preds = probs.argmax(axis=-1).copy()
    for pos in range(len(preds)):
        label = id2label[int(preds[pos])]
        if label == 'O':
            continue
        floor = entity_floors.get(label, 0.0)
        if probs[pos, int(preds[pos])] < floor:
            preds[pos] = 0
    return [id2label[int(p)] for p in preds]