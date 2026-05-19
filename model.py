"""
LSTM model architecture for multi-horizon time-series regression.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_lstm_model(
    lookback: int,
    n_features: int,
    horizon_steps: int,
    lstm_units: tuple[int, ...] = config.LSTM_UNITS,
    dropout: float = config.DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """
    Stacked LSTM encoder with multi-step linear output head.

    Architecture choices
    --------------------
    - Stacked LSTMs: capture short- and medium-term temporal dependencies.
    - return_sequences=True on first layer: passes full sequence to second LSTM.
    - Dropout: regularization to reduce overfitting on noisy load data.
    - Dense output (horizon_steps): direct multi-step regression in scaled space.
    - Linear activation: regression target (MW load); inverse-transform after predict.
    - MSE loss: standard for continuous forecasting; penalizes large errors.
    - MAE metric: interpretable in same units after inverse scaling.
    """
    inputs = keras.Input(shape=(lookback, n_features), name="sequence_input")

    x = inputs
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            name=f"lstm_{i+1}",
        )(x)
        x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)

    # Compress final temporal state
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)

    # Multi-step forecast vector
    outputs = layers.Dense(
        horizon_steps,
        activation="linear",
        name="forecast",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_load_forecaster")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_callbacks(model_path: str, patience: int = config.PATIENCE) -> list:
    """Training callbacks: early stopping and checkpoint best validation model."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
