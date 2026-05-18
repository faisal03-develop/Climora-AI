"""
Model evaluation: metrics and prediction visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

import config
from data import build_datasets


def inverse_transform_predictions(
    y_scaled: np.ndarray, target_scaler
) -> np.ndarray:
    """Inverse MinMax scaling for each step in the horizon dimension."""
    n_samples, horizon = y_scaled.shape
    flat = y_scaled.reshape(-1, 1)
    inv = target_scaler.inverse_transform(flat).reshape(n_samples, horizon)
    return inv


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Regression metrics on flattened multi-step forecasts.

    MAE and RMSE are in original MW units; R2 indicates variance explained.
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def evaluate_model(
    model_path: str | Path,
    horizon: str = config.DEFAULT_HORIZON,
    data: dict | None = None,
) -> dict:
    """
    Evaluate saved model on held-out test set.

    Returns metrics dict and saves prediction plots.
    """
    if data is None:
        data = build_datasets(horizon=horizon)

    model = keras.models.load_model(model_path)
    X_test, y_test = data["X_test"], data["y_test"]

    if len(X_test) == 0:
        return {"error": "Empty test set for this horizon"}

    y_pred_scaled = model.predict(X_test, verbose=0)
    target_scaler = data["target_scaler"]

    y_true = inverse_transform_predictions(y_test, target_scaler)
    y_pred = inverse_transform_predictions(y_pred_scaled, target_scaler)

    metrics = compute_metrics(y_true, y_pred)
    metrics["horizon"] = horizon
    metrics["n_test_samples"] = len(X_test)

    metrics_path = config.OUTPUT_DIR / f"metrics_{horizon}.json"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _plot_predictions(y_true, y_pred, horizon)
    _plot_error_distribution(y_true, y_pred, horizon)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, horizon: str) -> None:
    """Plot first forecast step and a sample multi-step trajectory."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    n_show = min(500, len(y_true))
    ax.plot(y_true[:n_show, 0], label="Actual (step 1)", linewidth=1)
    ax.plot(y_pred[:n_show, 0], label="Predicted (step 1)", linewidth=1, alpha=0.8)
    ax.set_title(f"Test Predictions vs Actual — horizon={horizon} (1st step)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"predictions_{horizon}.png", dpi=120)
    plt.close(fig)

    idx = len(y_true) // 2
    steps = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(steps), y_true[idx], "o-", label="Actual")
    ax.plot(range(steps), y_pred[idx], "s-", label="Predicted")
    ax.set_title(f"Single-sample multi-step forecast ({steps} steps)")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"multi_step_sample_{horizon}.png", dpi=120)
    plt.close(fig)


def _plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, horizon: str) -> None:
    errors = (y_true - y_pred).ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=60, edgecolor="black")
    ax.set_title(f"Prediction Error Distribution — {horizon}")
    ax.set_xlabel("Error (MW)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"errors_{horizon}.png", dpi=120)
    plt.close(fig)


def interpret_metrics(metrics: dict) -> str:
    """Plain-language performance summary."""
    mae, rmse, r2 = metrics["MAE"], metrics["RMSE"], metrics["R2"]
    lines = [
        f"MAE: {mae:,.2f} MW — average absolute forecast error.",
        f"RMSE: {rmse:,.2f} MW — penalizes large errors more than MAE.",
        f"R²: {r2:.4f} — proportion of variance explained (1.0 is perfect).",
    ]
    if r2 > 0.9:
        lines.append("Model explains most variance; suitable for operational use with monitoring.")
    elif r2 > 0.7:
        lines.append("Moderate fit; consider feature engineering or longer training.")
    else:
        lines.append("Weak fit for this horizon; try resampling, tuning, or shorter horizon.")
    return "\n".join(lines)
