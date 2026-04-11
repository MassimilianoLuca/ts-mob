"""
Step 4: Fine-Tune TimesFM with Covariates

Fine-tunes the pre-trained TimesFM 2.5 model for OD flow prediction
using attractiveness covariates from the gravity block and exogenous
signals (weather, calendar).

From the paper (Section 4.3 & Appendix B.3):
  - Optimizer: SGD, lr=5e-4, momentum=0.9
  - Linear warmup: 25 epochs
  - Total epochs: 50
  - Gradient clipping: 1.0
  - Input length: L=32
  - Loss: MSE
  - Covariates: [attractiveness, precipitation, temperature, hour_of_day]

Usage:
    python scripts/04_finetune_timesfm.py --config configs/default.yaml --data_dir data/synthetic

Note: This script requires TimesFM to be installed:
    pip install timesfm[xreg]
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, set_seed, get_device
from src.timesfm_finetune import (
    load_timesfm_model,
    prepare_temporal_data,
    finetune_timesfm,
)


def main():
    parser = argparse.ArgumentParser(description="Step 4: Fine-tune TimesFM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 1234))
    device = get_device(config)

    data_dir = args.data_dir or os.path.join(
        config["data"]["data_dir"],
        config["data"]["dataset"],
    )

    timesfm_cfg = config.get("timesfm", {})
    results_dir = config.get("output", {}).get("results_dir", "results")

    print("=" * 60)
    print("  Step 4: Fine-Tune TimesFM")
    print("=" * 60)

    # Load preprocessed data
    processed_path = os.path.join(data_dir, "processed", "preprocessed.pkl")
    with open(processed_path, "rb") as f:
        preprocessed = pickle.load(f)

    od_time_series = preprocessed["od_time_series"]
    weather_data = preprocessed["weather_data"]
    attractiveness = preprocessed.get("attractiveness", {})
    test_start_idx = preprocessed["test_start_idx"]
    n_timesteps = preprocessed["n_timesteps"]

    print(f"  OD pairs: {len(od_time_series)}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Attractiveness entries: {len(attractiveness)}")
    print(f"  Test start index: {test_start_idx}")

    # Build attractiveness time series (static value repeated)
    attr_time_series = {}
    for (o, d), val in attractiveness.items():
        attr_time_series[(o, d)] = np.full(n_timesteps, val, dtype=np.float32)

    # Use only training data for fine-tuning
    input_length = config["data"].get("input_length", 32)
    horizon = config["data"].get("horizon", 1)

    # Prepare training temporal data (up to test_start_idx)
    print(f"\n  Preparing temporal training data (L={input_length}, H={horizon})...")
    train_od_ts = {k: v[:test_start_idx] for k, v in od_time_series.items()}
    train_attr_ts = {k: v[:test_start_idx] for k, v in attr_time_series.items()}
    train_weather = {
        k: v[:test_start_idx] for k, v in weather_data.items()
    }

    contexts, covariates, targets = prepare_temporal_data(
        od_time_series=train_od_ts,
        attractiveness=train_attr_ts,
        weather_data=train_weather,
        input_length=input_length,
        horizon=horizon,
        exclude_self=config["data"].get("exclude_self_flows", True),
    )

    print(f"  Training samples: {len(contexts)}")

    if len(contexts) == 0:
        print("\n  WARNING: No training samples. Try increasing n_timesteps or decreasing input_length.")
        return

    # Load TimesFM model
    print(f"\n  Loading TimesFM: {timesfm_cfg.get('model_name', 'google/timesfm-2.5-200m-pytorch')}")
    model = load_timesfm_model(timesfm_cfg.get("model_name", "google/timesfm-2.5-200m-pytorch"))

    # Fine-tune
    save_path = os.path.join(results_dir, "timesfm_finetuned.pt")
    history = finetune_timesfm(
        model=model,
        contexts=contexts,
        covariates=covariates,
        targets=targets,
        config=timesfm_cfg,
        device=device,
        save_path=save_path,
    )

    print(f"\n  Final training loss: {history['loss'][-1]:.6f}")
    print("\n  Step 4 complete!")


if __name__ == "__main__":
    main()
