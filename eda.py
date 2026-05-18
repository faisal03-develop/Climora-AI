"""
Exploratory Data Analysis for time-series load/weather forecasting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data import handle_missing_values, load_raw_data


def run_eda(output_dir: Path | None = None) -> dict:
    """
    Run full EDA pipeline and save plots + summary statistics.

    Returns
    -------
    dict
        Insights and statistics for downstream preprocessing decisions.
    """
    out_dir = output_dir or config.EDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    insights: dict = {}

    # Structure
    insights["shape"] = list(df.shape)
    insights["columns"] = list(df.columns)
    insights["date_range"] = [str(df.index.min()), str(df.index.max())]
    insights["frequency"] = pd.infer_freq(df.index[:1000]) or "15min (expected)"

    # Summary statistics
    desc = df.describe().to_dict()
    insights["summary_statistics"] = desc

    # Missing values
    missing = df.isnull().sum().to_dict()
    insights["missing_per_column"] = missing
    cleaned, missing_info = handle_missing_values(df)
    insights["missing_handling"] = missing_info

    # Outliers (IQR on target)
    target = config.TARGET_COL
    q1, q3 = df[target].quantile(0.25), df[target].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((df[target] < lower) | (df[target] > upper)).sum()
    insights["outlier_count_iqr"] = int(outliers)
    insights["outlier_bounds"] = {"lower": float(lower), "upper": float(upper)}

    # Duplicates / gaps
    expected_delta = pd.Timedelta(minutes=15)
    deltas = df.index.to_series().diff().dropna()
    gap_count = int((deltas != expected_delta).sum())
    insights["irregular_interval_count"] = gap_count

    # Save summary text
    summary_path = out_dir / "eda_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== EDA Summary ===\n\n")
        for key, val in insights.items():
            f.write(f"{key}: {val}\n")

    _plot_time_series(df, out_dir)
    _plot_distributions(df, out_dir)
    _plot_seasonality(df, out_dir)
    _plot_correlation(df, out_dir)
    _plot_missing_and_outliers(df, out_dir, lower, upper)

    insights["plots_dir"] = str(out_dir)
    insights["key_findings"] = _generate_findings(df, insights)
    return insights


def _generate_findings(df: pd.DataFrame, insights: dict) -> list[str]:
    """Human-readable EDA insights for modeling."""
    target = config.TARGET_COL
    findings = [
        f"Dataset spans {insights['date_range'][0]} to {insights['date_range'][1]} "
        f"with {insights['shape'][0]:,} observations at ~15-minute resolution.",
        f"Target variable: '{target}' (MW). Exogenous feature: day-ahead forecast.",
        "Strong daily seasonality expected in electricity load patterns.",
        f"Missing values after load: {sum(insights['missing_per_column'].values())} "
        f"(handled via time interpolation + ffill/bfill).",
        f"IQR outliers on target: {insights['outlier_count_iqr']} "
        "(retained; likely valid peak load events).",
        f"Irregular time gaps: {insights['irregular_interval_count']}.",
        "Recommendation: MinMax scaling, cyclical calendar features, "
        "chronological 70/15/15 split, 7-day lookback for short horizons.",
    ]
    if config.FEATURE_COLS[0] in df.columns:
        corr = df[target].corr(df[config.FEATURE_COLS[0]])
        findings.append(
            f"Target vs day-ahead forecast correlation: {corr:.4f} — "
            "include forecast as input feature."
        )
    return findings


def _plot_time_series(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    sample = df.iloc[:: max(1, len(df) // 5000)]
    ax.plot(sample.index, sample[config.TARGET_COL], linewidth=0.6, label="Actual")
    if config.FEATURE_COLS[0] in df.columns:
        ax.plot(
            sample.index,
            sample[config.FEATURE_COLS[0]],
            linewidth=0.6,
            alpha=0.7,
            label="Day-ahead forecast",
        )
    ax.set_title("Total Load Time Series (sampled)")
    ax.set_xlabel("Time")
    ax.set_ylabel("MW")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_time_series.png", dpi=120)
    plt.close(fig)


def _plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df[config.TARGET_COL].hist(bins=50, ax=axes[0], edgecolor="black")
    axes[0].set_title(f"Distribution: {config.TARGET_COL}")
    if config.FEATURE_COLS[0] in df.columns:
        df[config.FEATURE_COLS[0]].hist(bins=50, ax=axes[1], edgecolor="black")
        axes[1].set_title(f"Distribution: {config.FEATURE_COLS[0]}")
    fig.tight_layout()
    fig.savefig(out_dir / "02_distributions.png", dpi=120)
    plt.close(fig)


def _plot_seasonality(df: pd.DataFrame, out_dir: Path) -> None:
    hourly = df.groupby(df.index.hour)[config.TARGET_COL].mean()
    monthly = df.groupby(df.index.month)[config.TARGET_COL].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hourly.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Average Load by Hour of Day")
    axes[0].set_xlabel("Hour")
    monthly.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Average Load by Month")
    axes[1].set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(out_dir / "03_seasonality.png", dpi=120)
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "04_correlation.png", dpi=120)
    plt.close(fig)


def _plot_missing_and_outliers(
    df: pd.DataFrame, out_dir: Path, lower: float, upper: float
) -> None:
    target = config.TARGET_COL
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df.index, df[target], s=1, alpha=0.3, label="Data")
    ax.axhline(lower, color="red", linestyle="--", label="IQR bounds")
    ax.axhline(upper, color="red", linestyle="--")
    ax.set_title("Outlier Bounds (IQR method)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_outliers.png", dpi=120)
    plt.close(fig)
