"""Run folds 4-5 (or 3-5) locally. Edit START_FOLD below."""

import os, sys, warnings, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold

from src.data.essay_dataset import EssayDataset
from src.models.essay_model import EssayScorer
from src.evaluation.essay_metrics import quadratic_weighted_kappa, mean_squared_error
from src.utils.config import load_config

# ── CONFIG ────────────────────────────────────────────────────
# fold 3 missing → START_FOLD = 2
# fold 3 exists  → START_FOLD = 3
START_FOLD  = 3        # 0-indexed: 3 = fold 4, 4 = fold 5
END_FOLD    = 5        # exclusive: runs up to fold END_FOLD
BATCH_SIZE  = 8        # safe for most GPUs; increase to 16 if ≥10GB VRAM
# ─────────────────────────────────────────────────────────────

cfg = load_config('configs/config.yaml')['stage2']
cfg['batch_size'] = BATCH_SIZE

torch.manual_seed(cfg['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Running folds {START_FOLD+1} to {END_FOLD}")

df        = pd.read_csv(cfg['train_path'])
tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
print(f"Data: {df.shape}")

skf       = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True,
                             random_state=cfg['seed'])
all_folds = list(skf.split(df, df['score']))

oof_preds = np.full(len(df), np.nan)
fold_qwks = []

for fold_idx in range(START_FOLD, END_FOLD):
    train_idx, val_idx = all_folds[fold_idx]
    print(f"\n{'='*50}")
    print(f"FOLD {fold_idx+1} / {cfg['n_folds']}")
    print(f"{'='*50}")
    print(f"{len(train_idx)} train | {len(val_idx)} val", flush=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(
        EssayDataset(train_df, tokenizer, cfg['max_length']),
        batch_size=cfg['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        EssayDataset(val_df, tokenizer, cfg['max_length']),
        batch_size=cfg['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True)

    model = EssayScorer(cfg['model_name'], cfg['dropout'])
    model.to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer    = AdamW(model.parameters(), lr=cfg['lr'],
                         weight_decay=cfg['weight_decay'])
    total_steps  = len(train_loader) * cfg['epochs']
    warmup_steps = int(total_steps * cfg['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()

    best_qwk, best_preds = -1.0, None
    fold_save_dir = f"outputs/models/essay/fold_{fold_idx+1}"
    os.makedirs(fold_save_dir, exist_ok=True)

    for epoch in range(cfg['epochs']):
        print(f"\n--- Epoch {epoch+1}/{cfg['epochs']} ---", flush=True)

        # Train
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["labels"].to(device)
            ttids = batch.get("token_type_ids")
            if ttids is not None:
                ttids = ttids.to(device)
            with autocast():
                loss, _ = model(ids, mask, ttids, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}", flush=True)

        print(f"Train loss: {total_loss/len(train_loader):.4f} | "
              f"Running eval...", flush=True)

        # Eval
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                ttids = batch.get("token_type_ids")
                if ttids is not None:
                    ttids = ttids.to(device)
                with autocast():
                    _, logits = model(ids, mask, ttids)
                all_p.extend(logits.cpu().numpy())
                all_l.extend(batch["labels"].numpy())

        all_p = np.array(all_p)
        all_l = np.array(all_l)
        qwk   = quadratic_weighted_kappa(all_l, all_p)
        mse   = mean_squared_error(all_l, all_p)
        print(f"Val QWK: {qwk:.4f} | Val MSE: {mse:.4f}", flush=True)

        if qwk > best_qwk:
            best_qwk, best_preds = qwk, all_p.copy()
            model.save_scorer(fold_save_dir, tokenizer)
            print(f"  → Best. Saved to {fold_save_dir}", flush=True)

    oof_preds[val_idx] = best_preds
    fold_qwks.append(best_qwk)
    print(f"\nFold {fold_idx+1} best QWK: {best_qwk:.4f}", flush=True)

    # Save OOF for this fold immediately
    pd.DataFrame({
        'essay_id':         df.iloc[val_idx]['essay_id'].values,
        'true_score':       df.iloc[val_idx]['score'].values,
        'oof_pred_raw':     best_preds,
        'oof_pred_rounded': np.clip(np.round(best_preds), 1, 6).astype(int),
        'fold':             fold_idx + 1,
    }).to_csv(f"outputs/models/essay/oof_fold{fold_idx+1}.csv", index=False)
    print(f"OOF fold {fold_idx+1} saved.", flush=True)

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

print(f"\nDone. Per-fold QWKs: {[round(q,4) for q in fold_qwks]}")