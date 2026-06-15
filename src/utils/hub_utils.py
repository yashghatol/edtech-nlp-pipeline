# src/utils/hub_utils.py
"""Resolve a model file path whether model_dir is a local folder or an HF Hub repo ID."""

import os
from huggingface_hub import hf_hub_download


def resolve_model_file(model_dir: str, filename: str) -> str:
    """Return a local filesystem path to `filename` inside `model_dir`.

    If model_dir is a local directory containing the file, return that path
    directly. Otherwise, treat model_dir as an HF Hub repo ID and download
    the file (cached after first download).
    """
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=model_dir, filename=filename)