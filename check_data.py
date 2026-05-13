import pandas as pd
import json
import os

# Stage 1: PII dataset
pii_path = r"data\raw\pii\train.json"

if not os.path.exists(pii_path):
    print(f"ERROR: File not found: {pii_path}")
else:
    with open(pii_path, "r", encoding="utf-8") as f:
        pii_data = json.load(f)
    print(f"PII dataset: {len(pii_data)} documents")
    print(f"Keys in first document: {list(pii_data[0].keys())}")

# Stage 2: Essay dataset
essay_path = r"data\raw\essay\train.csv"

if not os.path.exists(essay_path):
    print(f"ERROR: File not found: {essay_path}")
else:
    essay = pd.read_csv(essay_path)
    print(f"\nEssay dataset: {essay.shape[0]} rows x {essay.shape[1]} cols")
    print(f"Columns: {essay.columns.tolist()}")
    print(f"\nScore distribution:")
    print(essay["score"].value_counts().sort_index())

# Leakage check
if 'pii_data' in dir() and 'essay' in dir():
    pii_ids   = set(str(doc["document"]) for doc in pii_data)
    essay_ids = set(essay["essay_id"].astype(str))
    overlap   = pii_ids & essay_ids
    print(f"\n[LEAKAGE CHECK] Shared IDs: {len(overlap)}")
    if len(overlap) == 0:
        print("OK: No overlap found.")
    else:
        print("WARNING: Overlap found — investigate before Phase 1.")

print("\nDone.")