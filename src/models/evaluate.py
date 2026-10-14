"""
Model Evaluation Metrics and Founder-Facing Interpretations.
Computes RMSE, MAE, R², MAPE, MedianAE, and business-focused tolerance bands.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous technical and business metrics for regression predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Standard metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))

    # MAPE (avoiding division by zero)
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])))

    # Plain-language business metrics for founders:
    # 1. Percent of listings predicted within +- 5 Lakhs
    abs_errors = np.abs(y_true - y_pred)
    within_5_lakhs = float(np.mean(abs_errors <= 5.0) * 100.0)
    # 2. Percent of listings predicted within +- 10 Lakhs
    within_10_lakhs = float(np.mean(abs_errors <= 10.0) * 100.0)
    # 3. Percent of listings predicted within +- 10% relative error
    rel_errors = abs_errors / np.where(y_true == 0, 1e-6, y_true)
    within_10_percent = float(np.mean(rel_errors <= 0.10) * 100.0)

    founder_summary = (
        f"Typical price error is ±{mae:.2f} Lakhs (Median error: ±{medae:.2f} Lakhs). "
        f"{within_10_percent:.1f}% of properties are predicted within ±10% of asking price, "
        f"and {within_5_lakhs:.1f}% are within ±5 Lakhs."
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "median_ae": medae,
        "within_5_lakhs_pct": within_5_lakhs,
        "within_10_lakhs_pct": within_10_lakhs,
        "within_10_percent_pct": within_10_percent,
        "founder_summary": founder_summary
    }
