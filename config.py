"""
Configuration for LSTM time-series forecasting pipeline.
Adjust paths and hyperparameters here for different datasets.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "GUI_TOTAL_LOAD_DAYAHEAD_202412312300-202512312300.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
EDA_DIR = OUTPUT_DIR / "eda"

# Column mapping (adapt for other CSV schemas)
TIME_COL = "MTU (CET/CEST)"
AREA_COL = "Area"
TARGET_COL = "Actual Total Load (MW)"
FEATURE_COLS = ["Day-ahead Total Load Forecast (MW)"]

# Resampling: native data is 15-minute intervals (96 per day)
INTERVALS_PER_DAY = 96
INTERVALS_PER_HOUR = 4

# Forecast horizons in number of native (15-min) steps
# 1 day=96, 7 days=672, ~30 days month=2880, ~365 days year=35040
HORIZON_STEPS = {
    "1d": 96,       # 1 day @ 15-min
    "7d": 672,      # 7 days @ 15-min
    "month": 720,   # ~30 days ahead @ hourly (after resample)
    "year": 7,      # 7 days @ daily (full annual needs multi-year history)
}

# Lookback windows per horizon (in steps at chosen frequency)
# Shorter horizons use 15-min data; longer horizons use resampled series
LOOKBACK_STEPS = {
    "1d": 672,      # 7 days of 15-min history
    "7d": 2016,     # 21 days of 15-min history
    "month": 336,   # 14-day lookback @ hourly (fits validation split)
    "year": 30,     # 30 days daily (after resample)
}

RESAMPLE_RULE = {
    "1d": None,       # native 15-min
    "7d": None,
    "month": "1h",    # hourly for month-ahead
    "year": "1D",     # daily for year-ahead
}

# Train / val / test split (chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model & training
DEFAULT_HORIZON = "1d"
LSTM_UNITS = (64, 32)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 10
RANDOM_SEED = 42

# Set to an integer to use only the most recent N rows (faster experiments).
# Use None for the full dataset in production training.
MAX_ROWS = None

# Primary target for reporting
PRIMARY_TARGET = TARGET_COL
