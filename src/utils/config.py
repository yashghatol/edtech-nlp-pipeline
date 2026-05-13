"""Load and validate the master YAML config. Import this in every src/ module."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load master config YAML and return as nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_label_maps(cfg: dict) -> tuple[dict, dict]:
    """Return (label2id, id2label) dicts from Stage 1 config."""
    label2id = cfg["stage1_pii"]["model"]["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label
