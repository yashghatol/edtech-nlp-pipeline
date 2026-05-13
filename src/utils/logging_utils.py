"""Append rows to the experiment log CSV. Call this after every training run."""

import csv
import os
from datetime import date
from pathlib import Path


LOG_COLUMNS = [
    "experiment_id", "date", "model", "stage", "hypothesis",
    "change_made", "metric_before", "metric_after", "verdict", "notes",
]


def log_experiment(
    log_path: str,
    experiment_id: str,
    model: str,
    stage: str,
    hypothesis: str,
    change_made: str,
    metric_before: float | str,
    metric_after: float | str,
    verdict: str,
    notes: str = "",
) -> None:
    """Append one experiment row to the CSV log. Creates file with header if missing."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_id": experiment_id,
            "date":          str(date.today()),
            "model":         model,
            "stage":         stage,
            "hypothesis":    hypothesis,
            "change_made":   change_made,
            "metric_before": metric_before,
            "metric_after":  metric_after,
            "verdict":       verdict,
            "notes":         notes,
        })
    print(f"[experiment_log] Logged {experiment_id} → {log_path}")
