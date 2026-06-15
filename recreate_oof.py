"""Recreate OOF predictions for folds whose oof_foldX.csv is missing."""

import os, sys, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

from src.data.essay_dataset import EssayDataset
from src.models.essay_model import EssayScorer
from src.evaluation.essay_metrics import quadratic_weighted_kappa
from src.utils.config import load_config

cfg    = load_config('configs/config.yaml')['stage2']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

df        = pd.read_csv(cfg['train_path'])
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg['seed'])
all_folds = list(skf.split(df, df['score']))

for fold_idx in range(5):
    oof_path   = f"outputs/models/essay/oof_fold{fold_idx+1}.csv"
    model_path = f"outputs/models/essay/fold_{fold_idx+1}"

    if os.path.exists(oof_path):
        print(f"Fold {fold_idx+1}: OOF already exists, skipping.")
        continue

    # Accept either safetensors or pytorch_model.bin
    has_model = (
        os.path.exists(os.path.join(model_path, 'model.safetensors')) or
        os.path.exists(os.path.join(model_path, 'pytorch_model.bin'))
    )
    if not has_model:
        print(f"Fold {fold_idx+1}: model MISSING, cannot recreate OOF.")
        continue

    print(f"Fold {fold_idx+1}: recreating OOF...", flush=True)
    _, val_idx = all_folds[fold_idx]
    val_df     = df.iloc[val_idx].reset_index(drop=True)

    tokenizer  = AutoTokenizer.from_pretrained("outputs/models/essay/fold_1")
    val_loader = DataLoader(
        EssayDataset(val_df, tokenizer, cfg['max_length']),
        batch_size=8, shuffle=False, num_workers=0)

    model = EssayScorer(model_path, cfg['dropout'])
    model.regressor.load_state_dict(
        torch.load(
            os.path.join(model_path, 'regressor_head.pt'),
            map_location=device
        )
    )
    model.to(device).eval()

    all_p = []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with autocast():
                _, logits = model(ids, mask)
            all_p.extend(logits.cpu().numpy())

    all_p = np.array(all_p)
    qwk   = quadratic_weighted_kappa(val_df['score'].values, all_p)
    print(f"  QWK: {qwk:.4f}", flush=True)

    pd.DataFrame({
        'essay_id':         val_df['essay_id'].values,
        'true_score':       val_df['score'].values,
        'oof_pred_raw':     all_p,
        'oof_pred_rounded': np.clip(np.round(all_p), 1, 6).astype(int),
        'fold':             fold_idx + 1,
    }).to_csv(oof_path, index=False)
    print(f"  Saved to {oof_path}", flush=True)

    del model
    torch.cuda.empty_cache()

print("\nDone.")