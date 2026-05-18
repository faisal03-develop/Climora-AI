"""
Main entry point: EDA → train → evaluate → inference demo.

Usage:
    python main.py --stage all --horizon 1d
    python main.py --stage eda
    python main.py --stage train --horizon 7d
    python main.py --stage evaluate --horizon 1d
    python main.py --stage predict --horizon 1d
    python main.py --stage all --horizon all
"""

from __future__ import annotations

import argparse
import json
import sys

import config
from eda import run_eda
from evaluate import evaluate_model, interpret_metrics
from inference import predict_forecast
from train import train_horizon


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM load forecasting pipeline")
    parser.add_argument(
        "--stage",
        choices=["eda", "train", "evaluate", "predict", "all"],
        default="all",
    )
    parser.add_argument(
        "--horizon",
        default=config.DEFAULT_HORIZON,
        help="Forecast horizon: 1d, 7d, month, year, or 'all'",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use recent 12k rows and 12 epochs for a quicker run",
    )
    return parser.parse_args()


def horizons_to_run(horizon_arg: str) -> list[str]:
    if horizon_arg == "all":
        return list(config.HORIZON_STEPS.keys())
    if horizon_arg not in config.HORIZON_STEPS:
        raise ValueError(f"Unknown horizon: {horizon_arg}")
    return [horizon_arg]


def main():
    args = parse_args()
    if args.fast:
        config.MAX_ROWS = 12000
        config.EPOCHS = 12
    horizons = horizons_to_run(args.horizon)

    if args.stage in ("eda", "all"):
        print("=== Exploratory Data Analysis ===")
        insights = run_eda()
        print("\n".join(insights["key_findings"]))
        print(f"\nEDA plots saved to: {insights['plots_dir']}")

    for horizon in horizons:
        if args.stage in ("train", "all"):
            print(f"\n=== Training LSTM (horizon={horizon}) ===")
            try:
                result = train_horizon(horizon=horizon)
                print(f"Best model saved: {result['model_path']}")
            except Exception as exc:
                print(f"Training failed for {horizon}: {exc}", file=sys.stderr)
                if args.horizon != "all":
                    raise
                continue

        if args.stage in ("evaluate", "all"):
            model_path = config.MODEL_DIR / f"lstm_{horizon}_best.keras"
            if not model_path.exists():
                print(f"Skipping evaluate for {horizon}: no model found.")
                continue
            print(f"\n=== Evaluation (horizon={horizon}) ===")
            eval_result = evaluate_model(model_path, horizon=horizon)
            if "error" in eval_result:
                print(eval_result["error"])
                continue
            metrics = eval_result["metrics"]
            print(json.dumps(metrics, indent=2))
            print(interpret_metrics(metrics))

        if args.stage in ("predict", "all"):
            print(f"\n=== Inference demo (horizon={horizon}) ===")
            try:
                out = predict_forecast(horizon=horizon)
                print(json.dumps(
                    {
                        "horizon": out["horizon"],
                        "last_timestamp": out["last_timestamp"],
                        "first_5_steps_mw": out["forecast_mw"][:5],
                        "total_steps": out["forecast_steps"],
                    },
                    indent=2,
                ))
            except Exception as exc:
                print(f"Inference failed for {horizon}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
