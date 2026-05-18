"""
Production inference: load model and generate real-time forecasts.

Deployment considerations (summary)
-----------------------------------
1. Data pipeline: ingest 15-min CSV/API rows, parse timestamps, impute gaps,
   engineer calendar features, scale with persisted joblib scalers.
2. Model versioning: store `{horizon}_best.keras` + scalers + metadata.json
   under versioned paths (e.g. models/v1.2.0/); pin version in serving config.
3. Monitoring: track MAE/RMSE on rolling holdout, data drift (distribution of load),
   and latency; alert if error exceeds baseline by >20%.
4. Serialization: Keras native .keras format; scalers via joblib; metadata JSON.
5. Serving: batch offline forecasts via `predict_forecast()`; for API serving,
   wrap in FastAPI/Flask with health check and model warm-load on startup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

import config
from data import (
    create_sequences,
    load_artifacts,
    load_raw_data,
    prepare_series,
    scale_dataframe,
    handle_missing_values,
)
from evaluate import inverse_transform_predictions


class LoadForecastService:
    """
    Production-style wrapper for loading artifacts and serving forecasts.
    """

    def __init__(self, horizon: str = config.DEFAULT_HORIZON, model_version: str | None = None):
        self.horizon = horizon
        self.model_path = config.MODEL_DIR / f"lstm_{horizon}_best.keras"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run training first."
            )
        self.model = keras.models.load_model(self.model_path)
        artifacts = load_artifacts(horizon)
        self.feature_scaler = artifacts["feature_scaler"]
        self.target_scaler = artifacts["target_scaler"]
        self.metadata = artifacts["metadata"]
        self.lookback = self.metadata["lookback"]
        self.horizon_steps = self.metadata["horizon_steps"]
        self.feature_cols = self.metadata["feature_cols"]

    def predict_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate multi-step forecast from recent historical DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least `lookback` rows with required feature columns.

        Returns
        -------
        np.ndarray
            Shape (horizon_steps,) in original MW units.
        """
        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows; got {len(df)}")

        scaled = scale_dataframe(
            df.iloc[-self.lookback :],
            self.feature_cols,
            config.TARGET_COL,
            self.feature_scaler,
            self.target_scaler,
        )
        features = scaled[self.feature_cols].values
        X = features.reshape(1, self.lookback, len(self.feature_cols))
        y_scaled = self.model.predict(X, verbose=0)
        return inverse_transform_predictions(y_scaled, self.target_scaler)[0]

    def predict_latest_from_csv(self, csv_path: Path | None = None) -> dict:
        """End-to-end: load CSV, preprocess, return next-horizon forecast."""
        raw = load_raw_data(csv_path)
        cleaned, _ = handle_missing_values(raw)
        series, _ = prepare_series(cleaned, self.horizon)
        forecast = self.predict_from_dataframe(series)
        return {
            "horizon": self.horizon,
            "forecast_steps": self.horizon_steps,
            "forecast_mw": forecast.tolist(),
            "last_timestamp": str(series.index[-1]),
        }


def predict_forecast(horizon: str = config.DEFAULT_HORIZON) -> dict:
    """Convenience function for CLI / scheduled jobs."""
    service = LoadForecastService(horizon=horizon)
    return service.predict_latest_from_csv()
