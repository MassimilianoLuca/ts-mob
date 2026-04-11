"""
Step 1: Prepare Data

Loads raw mobility data and prepares it for the TS-Mob pipeline:
  - Loads OD flows, tile features, centroids, and weather data
  - Applies log transform to population
  - Performs chronological train/test split
  - Builds OD time series for temporal training
  - Saves preprocessed data for subsequent steps

Usage:
    python scripts/01_prepare_data.py --config configs/default.yaml --data_dir data/synthetic
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, set_seed
from src.data_loader import load_od_tensor, build_o2d2flow, temporal_train_test_split


def main():
    parser = argparse.ArgumentParser(description="Step 1: Prepare data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 1234))

    data_dir = args.data_dir or os.path.join(
        config["data"]["data_dir"],
        config["data"]["dataset"],
    )

    print("=" * 60)
    print("  Step 1: Prepare Data")
    print("=" * 60)
    print(f"  Data directory: {data_dir}")

    # Load data
    data = load_od_tensor(data_dir)

    flows_df = data["flows_df"]
    tile2features = data["tile2features"]
    tile2centroid = data["tile2centroid"]

    print(f"\n  Tiles: {len(tile2features)}")
    print(f"  Flow records: {len(flows_df)}")
    print(f"  Unique timestamps: {flows_df['timestamp'].nunique()}")

    # Chronological train/test split
    train_fraction = config["data"].get("train_fraction", 0.8)
    print(f"\n  Splitting data (train fraction: {train_fraction})...")
    train_df, test_df = temporal_train_test_split(flows_df, train_fraction)

    # Build OD time series (for temporal module)
    print("\n  Building OD time series...")
    timestamps = sorted(flows_df["timestamp"].unique())
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    n_timesteps = len(timestamps)

    tile_ids = sorted(tile2features.keys())
    od_time_series = {}
    for o in tile_ids:
        for d in tile_ids:
            if config["data"].get("exclude_self_flows", True) and o == d:
                continue
            od_time_series[(o, d)] = np.zeros(n_timesteps, dtype=np.float32)

    for _, row in flows_df.iterrows():
        o, d, ts, f = str(row["origin"]), str(row["destination"]), row["timestamp"], int(row["flow"])
        if (o, d) in od_time_series:
            od_time_series[(o, d)][ts_to_idx[ts]] += f

    print(f"  OD pairs: {len(od_time_series)}")
    print(f"  Time steps: {n_timesteps}")

    # Build aggregate flows for gravity model
    agg_train = train_df.groupby(["origin", "destination"])["flow"].sum().reset_index()
    o2d2flow_train = build_o2d2flow(agg_train)

    # Prepare weather data
    weather_data = None
    if "weather_df" in data:
        weather_df = data["weather_df"]
        weather_data = {
            "precipitation_mm": np.zeros(n_timesteps, dtype=np.float32),
            "temperature_c": np.zeros(n_timesteps, dtype=np.float32),
            "hour_of_day": np.zeros(n_timesteps, dtype=np.float32),
        }
        for _, row in weather_df.iterrows():
            ts = row["timestamp"]
            if ts in ts_to_idx:
                idx = ts_to_idx[ts]
                weather_data["precipitation_mm"][idx] = row["precipitation_mm"]
                weather_data["temperature_c"][idx] = row["temperature_c"]
                weather_data["hour_of_day"][idx] = row["hour_of_day"]
        print(f"  Weather data: loaded ({n_timesteps} timesteps)")

    # Compute test start index
    test_timestamps = sorted(test_df["timestamp"].unique())
    test_start_idx = ts_to_idx[test_timestamps[0]] if test_timestamps else n_timesteps

    # Save preprocessed data
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    preprocessed = {
        "tile_ids": tile_ids,
        "tile2features": tile2features,
        "tile2centroid": tile2centroid,
        "o2d2flow_train": o2d2flow_train,
        "od_time_series": od_time_series,
        "weather_data": weather_data,
        "timestamps": timestamps,
        "test_start_idx": test_start_idx,
        "n_timesteps": n_timesteps,
    }

    output_path = os.path.join(processed_dir, "preprocessed.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(preprocessed, f)

    print(f"\n  Preprocessed data saved to: {output_path}")
    print(f"  Test start index: {test_start_idx} / {n_timesteps}")
    print("\n  Step 1 complete!")


if __name__ == "__main__":
    main()
