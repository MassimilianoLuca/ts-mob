"""
Evaluation metrics for TS-Mob.

Implements the three metrics used in the paper:
- RMSE (Root Mean Squared Error)
- MAE  (Mean Absolute Error)
- CPC  (Common Part of Commuters)
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt( (1/n) * sum_i (y_true_i - y_pred_i)^2 )

    Args:
        y_true: Ground truth OD flows, shape (n,) or (n, m).
        y_pred: Predicted OD flows, same shape as y_true.

    Returns:
        RMSE value (lower is better).
    """
    return float(np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = (1/N) * sum_{i,j} |T_ij_hat - T_ij|

    Args:
        y_true: Ground truth OD flows.
        y_pred: Predicted OD flows.

    Returns:
        MAE value (lower is better).
    """
    return float(np.mean(np.abs(y_true.flatten() - y_pred.flatten())))


def cpc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Common Part of Commuters.

    CPC(T_hat, T) = 2 * sum_{i,j} min(T_hat_ij, T_ij) / (sum T_hat_ij + sum T_ij)

    CPC ranges between 0 and 1:
    - CPC = 0 means no flows in common
    - CPC = 1 means identical flow distributions

    Args:
        y_true: Ground truth OD flows.
        y_pred: Predicted OD flows.

    Returns:
        CPC value (higher is better).
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    numerator = 2.0 * np.sum(np.minimum(y_true_flat, y_pred_flat))
    denominator = np.sum(y_true_flat) + np.sum(y_pred_flat)

    if denominator > 0:
        return float(numerator / denominator)
    return 0.0


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth OD flows.
        y_pred: Predicted OD flows.

    Returns:
        Dictionary with RMSE, MAE, and CPC values.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "cpc": cpc(y_true, y_pred),
    }
