"""Evaluation metrics for essay scoring."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_score: int = 1,
    max_score: int = 6,
) -> float:
    """Quadratic Weighted Kappa — official AES 2.0 competition metric.

    Args:
        y_true: Ground truth integer scores.
        y_pred: Raw predicted scores (will be rounded and clipped).
        min_score: Minimum valid score.
        max_score: Maximum valid score.

    Returns:
        QWK in [-1, 1]. Higher is better; 1.0 = perfect agreement.
    """
    y_pred_clipped = np.clip(np.round(y_pred), min_score, max_score).astype(int)
    y_true = np.array(y_true).astype(int)
    return float(cohen_kappa_score(y_true, y_pred_clipped, weights="quadratic"))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE between raw predictions and true scores (no rounding).

    Args:
        y_true: Ground truth scores.
        y_pred: Raw predicted scores.

    Returns:
        MSE value (lower is better).
    """
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
if __name__ == "__main__":
    import numpy as np
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred = np.array([1.2, 2.4, 2.8, 4.1, 4.9, 5.7])

    print(f"QWK: {quadratic_weighted_kappa(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")