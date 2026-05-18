"""
Data loading, cleaning, feature engineering, and sequence generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


def parse_timestamp(mtu_series: pd.Series) -> pd.DatetimeIndex:
    """Parse MTU interval strings; use interval start as timestamp."""
    starts = mtu_series.str.split(" - ").str[0]
    # Strip DST timezone suffixes, e.g. "30/03/2025 01:30 (CET)"
    starts = starts.str.replace(r"\s*\((CET|CEST)\)\s*$", "", regex=True)
    return pd.to_datetime(starts, format="%d/%m/%Y %H:%M", dayfirst=True)


def load_raw_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load CSV and return a cleaned DataFrame indexed by timestamp.

    Returns
    -------
    pd.DataFrame
        Columns include target, optional features, and calendar encodings.
    """
    path = csv_path or config.DATA_PATH
    df = pd.read_csv(path)
    df["timestamp"] = parse_timestamp(df[config.TIME_COL])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")

    numeric_cols = [config.TARGET_COL] + config.FEATURE_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[numeric_cols].copy()
    df = _add_calendar_features(df)
    if config.MAX_ROWS is not None:
        df = df.iloc[-config.MAX_ROWS :]
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time features help LSTM capture seasonality."""
    hour = df.index.hour + df.index.minute / 60.0
    day_of_week = df.index.dayofweek
    day_of_year = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using forward-fill then backward-fill.
    Suitable for short gaps in regularly sampled series.
    """
    out = df.copy()
    missing_before = out.isnull().sum().sum()
    out = out.interpolate(method="time", limit_direction="both")
    out = out.ffill().bfill()
    missing_after = out.isnull().sum().sum()
    return out, {"missing_before": int(missing_before), "missing_after": int(missing_after)}


def resample_for_horizon(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Resample series when horizon requires coarser granularity."""
    rule = config.RESAMPLE_RULE.get(horizon)
    if rule is None:
        return df
    numeric = df.select_dtypes(include=[np.number])
    return numeric.resample(rule).mean().dropna()


def prepare_series(
    df: pd.DataFrame,
    horizon: str,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Select feature columns and target for modeling at a given horizon.
    """
    target = config.TARGET_COL
    feature_names = [target] + config.FEATURE_COLS + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    ]
    resampled = resample_for_horizon(df, horizon)
    available = [c for c in feature_names if c in resampled.columns]
    return resampled[available], available


def chronological_split(
    n_samples: int,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
) -> Tuple[slice, slice, slice]:
    """Return slice objects for train/val/test (no shuffle)."""
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit separate scalers for features and target (inverse-transform predictions)."""
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])
    return feature_scaler, target_scaler


def scale_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Return scaled copy; target stored in scaled_target column for sequences."""
    out = df.copy()
    out[feature_cols] = feature_scaler.transform(df[feature_cols])
    out["scaled_target"] = target_scaler.transform(df[[target_col]]).ravel()
    return out


def create_sequences(
    scaled_df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for multi-step forecasting.

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples, horizon_steps) — future target values
    """
    features = scaled_df[feature_cols].values
    target = scaled_df["scaled_target"].values
    n = len(scaled_df)
    max_start = n - lookback - horizon_steps + 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough data: need {lookback + horizon_steps} rows, have {n}"
        )

    X_list, y_list = [], []
    for i in range(max_start):
        X_list.append(features[i : i + lookback])
        y_list.append(target[i + lookback : i + lookback + horizon_steps])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_datasets(
    horizon: str = config.DEFAULT_HORIZON,
    csv_path: Path | None = None,
) -> dict:
    """
    Full preprocessing pipeline: load, clean, scale, sequence, split.

    Returns dict with arrays, scalers, metadata for training and inference.
    """
    raw = load_raw_data(csv_path)
    cleaned, missing_info = handle_missing_values(raw)
    series_df, feature_cols = prepare_series(cleaned, horizon)

    lookback = config.LOOKBACK_STEPS[horizon]
    horizon_steps = config.HORIZON_STEPS[horizon]

    n = len(series_df)
    train_sl, val_sl, test_sl = chronological_split(n)

    train_df = series_df.iloc[train_sl]
    val_df = series_df.iloc[val_sl]
    test_df = series_df.iloc[test_sl]

    feature_scaler, target_scaler = fit_scalers(
        train_df, feature_cols, config.TARGET_COL
    )

    train_scaled = scale_dataframe(
        train_df, feature_cols, config.TARGET_COL, feature_scaler, target_scaler
    )
    val_scaled = scale_dataframe(
        val_df, feature_cols, config.TARGET_COL, feature_scaler, target_scaler
    )
    test_scaled = scale_dataframe(
        test_df, feature_cols, config.TARGET_COL, feature_scaler, target_scaler
    )

    # Build sequences within each split to avoid temporal leakage across boundaries
    X_train, y_train = create_sequences(
        train_scaled, feature_cols, lookback, horizon_steps
    )
    X_val, y_val = create_sequences(
        val_scaled, feature_cols, lookback, horizon_steps
    )
    X_test, y_test = create_sequences(
        test_scaled, feature_cols, lookback, horizon_steps
    )

    metadata = {
        "horizon": horizon,
        "lookback": lookback,
        "horizon_steps": horizon_steps,
        "feature_cols": feature_cols,
        "target_col": config.TARGET_COL,
        "missing_info": missing_info,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "resample_rule": config.RESAMPLE_RULE.get(horizon),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "metadata": metadata,
        "series_df": series_df,
    }


def save_artifacts(artifacts: dict, horizon: str) -> Path:
    """Persist scalers and metadata for production inference."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = config.ARTIFACT_DIR / f"lstm_{horizon}"
    joblib.dump(artifacts["feature_scaler"], f"{prefix}_feature_scaler.joblib")
    joblib.dump(artifacts["target_scaler"], f"{prefix}_target_scaler.joblib")
    meta_path = Path(f"{prefix}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifacts["metadata"], f, indent=2)
    return prefix


def load_artifacts(horizon: str) -> dict:
    """Load scalers and metadata saved during training."""
    prefix = config.ARTIFACT_DIR / f"lstm_{horizon}"
    with open(f"{prefix}_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "feature_scaler": joblib.load(f"{prefix}_feature_scaler.joblib"),
        "target_scaler": joblib.load(f"{prefix}_target_scaler.joblib"),
        "metadata": metadata,
    }
