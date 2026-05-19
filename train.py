"""
Model training with early stopping and checkpointing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
from data import build_datasets, save_artifacts
from model import build_lstm_model, get_callbacks


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Reproducibility for experiments and deployment baselines."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_horizon(horizon: str = config.DEFAULT_HORIZON) -> dict:
    """
    Train LSTM for a specific forecast horizon.

    Returns
    -------
    dict
        history, model path, datasets, and metadata.
    """
    set_seed()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = build_datasets(horizon=horizon)
    meta = data["metadata"]
    lookback = meta["lookback"]
    horizon_steps = meta["horizon_steps"]
    n_features = data["X_train"].shape[2]

    if len(data["X_train"]) == 0:
        raise RuntimeError(f"No training sequences for horizon={horizon}")

    model = build_lstm_model(
        lookback=lookback,
        n_features=n_features,
        horizon_steps=horizon_steps,
    )

    model_path = str(config.MODEL_DIR / f"lstm_{horizon}_best.keras")
    callbacks = get_callbacks(model_path)

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    save_artifacts(
        {
            "feature_scaler": data["feature_scaler"],
            "target_scaler": data["target_scaler"],
            "metadata": meta,
        },
        horizon,
    )

    history_path = config.OUTPUT_DIR / f"training_history_{horizon}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    _plot_training_curves(history, horizon)

    return {
        "model": model,
        "model_path": model_path,
        "history": history.history,
        "data": data,
        "horizon": horizon,
    }


def _plot_training_curves(history, horizon: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (MSE)")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history.history["mae"], label="train")
    axes[1].plot(history.history["val_mae"], label="val")
    axes[1].set_title("MAE (scaled)")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"training_curves_{horizon}.png", dpi=120)
    plt.close(fig)
