# src/training/pii_trainer.py

import os
import math                   # 🔴 Required for log-scaled class weights
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from collections import Counter

from src.evaluation.pii_metrics import compute_entity_f1


def _labels_ordered(label2id: dict) -> list:
    """Return label strings sorted by their integer ID (matches logit dimension order)."""
    return [lbl for lbl, _ in sorted(label2id.items(), key=lambda x: x[1])]


def compute_class_weights(train_records: list, label2id: dict,
                           device: torch.device,
                           max_weight: float = 50.0,
                           min_weight: float = 0.1) -> torch.Tensor:
    """Log-scaled inverse-frequency class weights, capped at max_weight.

    🔴 CORRECTED from original guide: T0 revealed O = 99.95% of tokens.
    Raw inverse-frequency produces weights up to 384,615 for labels with 1
    example (B-STREET_ADDRESS: 2 samples → raw weight ~192,000). This
    destabilises training completely — gradients explode and entity_f1 ≈ 0.

    Log-scaling + capping compresses the range to [1.0, 50.0]:
      raw B-STREET_ADDRESS: ~192,000  →  log-scaled: 12.1  →  capped: 12.1
      raw I-URL_PERSONAL: ~384,615   →  log-scaled: 12.9  →  capped: 12.9
      raw O:               ~0.08     →  log-scaled: 1.0   (floor from 1+log(x))

    max_weight=50.0 is a safe ceiling for extreme labels with 1-2 examples.

    ⚠️ LEAKAGE: Pass train_records only. Never pass val_records or full records.
    """
    counter    = Counter()
    for rec in train_records:
        counter.update(rec["labels"])

    total      = sum(counter.values())
    num_labels = len(label2id)
    weights    = []

    for label in _labels_ordered(label2id):
        count = counter.get(label, 1)          # Default 1 avoids log(0)
        raw_w = total / (num_labels * count)   # Inverse frequency
        log_w = 1.0 + math.log(raw_w)          # Log dampens extreme values
        weights.append(max(min_weight, min(log_w, max_weight))) # Hard cap for stability

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader: DataLoader, optimizer,
                    scheduler, scaler: GradScaler,
                    device: torch.device,
                    class_weights: torch.Tensor = None) -> float:
    """Train one epoch. Returns mean loss across all batches."""
    model.train()
    loss_fn    = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            out    = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            loss   = loss_fn(
                logits.view(-1, model.num_labels),
                labels.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader: DataLoader, device: torch.device,
             id2label: dict, return_logits: bool = False) -> tuple:
    """Validation loop. Returns (all_true, all_pred) as List[List[str]] for seqeval.

    If return_logits=True, also returns List[np.ndarray] for threshold sweeping in T7.
    -100 positions are excluded — seqeval receives clean word-level sequences.
    """
    import numpy as np
    model.eval()
    all_true, all_pred, all_logits = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]         # CPU for indexing

            with autocast():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = out["logits"].cpu().float().numpy()  # (B, L, 13)
            preds  = logits.argmax(axis=-1)               # (B, L)

            for b_logits, pred_seq, true_seq in zip(logits, preds, labels.numpy()):
                word_mask    = true_seq != -100
                word_preds   = pred_seq[word_mask]
                word_true    = true_seq[word_mask]
                word_logits  = b_logits[word_mask]        # (num_words, 13)

                all_pred.append([id2label[int(p)] for p in word_preds])
                all_true.append([id2label[int(t)] for t in word_true])
                if return_logits:
                    all_logits.append(word_logits)

    if return_logits:
        return all_true, all_pred, all_logits
    return all_true, all_pred


def run_training(model, train_dataset, val_dataset, cfg: dict,
                 device: torch.device, tokenizer=None,
                 class_weights: torch.Tensor = None) -> float:
    """Full training loop with best-model checkpointing on entity F1.

    Infrastructure notes:
    - num_workers=0: avoids silent GPU hang (Phase 2 bug)
    - GradScaler: fp16, ~40% VRAM reduction
    - gradient_checkpointing: call model.enable_gradient_checkpointing() before this
    - Checkpoint on val improvement only
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,        # ⚠️ CRITICAL: ≥1 causes silent GPU hang
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01
    )
    total_steps  = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(cfg.get("warmup_fraction", 0.1) * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()
    id2label     = {v: k for k, v in train_dataset.label2id.items()}
    best_f1      = 0.0
    best_epoch   = 0
    save_dir     = cfg["model_save_dir"]

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, class_weights
        )
        all_true, all_pred = evaluate(model, val_loader, device, id2label)
        metrics   = compute_entity_f1(all_true, all_pred)
        entity_f1 = metrics["entity_f1"]

        print(f"Epoch {epoch}/{cfg['num_epochs']}  "
              f"loss={train_loss:.4f}  "
              f"entity_F1={entity_f1:.4f}  "
              f"P={metrics['entity_precision']:.4f}  "
              f"R={metrics['entity_recall']:.4f}")

        if entity_f1 > best_f1:
            best_f1    = entity_f1
            best_epoch = epoch
            os.makedirs(save_dir, exist_ok=True)
            model.backbone.save_pretrained(save_dir)
            torch.save(
                model.classifier.state_dict(),
                os.path.join(save_dir, "classifier_head.pt")
            )
            if tokenizer is not None:
                tokenizer.save_pretrained(save_dir)
            print(f"  → Saved best model (epoch {epoch})")

    print(f"\nBest entity F1: {best_f1:.4f} at epoch {best_epoch}")
    return best_f1