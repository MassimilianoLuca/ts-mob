"""
Data loading and dataset construction for TS-Mob.

Handles:
1. Loading raw mobility data (OD tensors, tile features, centroids)
2. FlowDataset for training the gravity model
3. TemporalDataset for fine-tuning TimesFM with covariates
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import haversine_distance


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_od_tensor(data_dir: str) -> dict:
    """
    Load all preprocessed data from a dataset directory.

    Expected directory structure:
        data_dir/
        ├── od_flows.csv           # Columns: timestamp, origin, destination, flow
        ├── tile_features.csv      # Columns: tile_id, population, poi_cat_1, ..., poi_cat_16
        ├── tile_centroids.csv     # Columns: tile_id, latitude, longitude
        ├── weather.csv            # Columns: timestamp, precipitation_mm, temperature_c
        └── attractiveness.npy     # (optional) Pre-computed attractiveness matrix

    Args:
        data_dir: Path to dataset directory.

    Returns:
        Dictionary with all loaded data structures.
    """
    data = {}

    # Load OD flows
    flows_path = os.path.join(data_dir, "od_flows.csv")
    if os.path.exists(flows_path):
        flows_df = pd.read_csv(flows_path, dtype={"origin": str, "destination": str})
        data["flows_df"] = flows_df

    # Load tile features (population + 16 POI categories)
    features_path = os.path.join(data_dir, "tile_features.csv")
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path, dtype={"tile_id": str})
        data["features_df"] = features_df

        # Build tile_id -> feature vector mapping
        tile_ids = features_df["tile_id"].values
        feature_cols = [c for c in features_df.columns if c != "tile_id"]
        data["tile2features"] = {
            tid: features_df.loc[features_df["tile_id"] == tid, feature_cols].values.flatten().tolist()
            for tid in tile_ids
        }

    # Load tile centroids
    centroids_path = os.path.join(data_dir, "tile_centroids.csv")
    if os.path.exists(centroids_path):
        centroids_df = pd.read_csv(centroids_path, dtype={"tile_id": str})
        data["centroids_df"] = centroids_df
        data["tile2centroid"] = {
            row["tile_id"]: (row["latitude"], row["longitude"])
            for _, row in centroids_df.iterrows()
        }

    # Load weather data
    weather_path = os.path.join(data_dir, "weather.csv")
    if os.path.exists(weather_path):
        data["weather_df"] = pd.read_csv(weather_path)

    # Load pre-computed attractiveness (if available)
    attr_path = os.path.join(data_dir, "attractiveness.npy")
    if os.path.exists(attr_path):
        data["attractiveness"] = np.load(attr_path, allow_pickle=True).item()

    return data


def build_o2d2flow(flows_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Build nested origin -> destination -> flow dictionary.

    Args:
        flows_df: DataFrame with columns [origin, destination, flow].

    Returns:
        Nested dict: o2d2flow[origin][destination] = flow_count.
    """
    o2d2flow = {}
    for _, row in flows_df.iterrows():
        o = str(row["origin"])
        d = str(row["destination"])
        f = int(row["flow"])
        if o not in o2d2flow:
            o2d2flow[o] = {}
        o2d2flow[o][d] = o2d2flow[o].get(d, 0) + f
    return o2d2flow


def temporal_train_test_split(
    flows_df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split.

    The paper uses time-based splits (last 20% for TaxiBJ, last 10 days
    for BikeNYC, last 3 weeks for Spain OD).

    Args:
        flows_df: DataFrame with a 'timestamp' column, sorted by time.
        train_fraction: Fraction of data for training.

    Returns:
        (train_df, test_df) tuple.
    """
    flows_df = flows_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(flows_df) * train_fraction)

    # Find the split timestamp
    timestamps = flows_df["timestamp"].unique()
    timestamps.sort()
    split_ts_idx = int(len(timestamps) * train_fraction)
    split_ts = timestamps[split_ts_idx]

    train_df = flows_df[flows_df["timestamp"] < split_ts]
    test_df = flows_df[flows_df["timestamp"] >= split_ts]

    print(f"Train: {len(train_df)} rows ({len(train_df['timestamp'].unique())} timesteps)")
    print(f"Test:  {len(test_df)} rows ({len(test_df['timestamp'].unique())} timesteps)")

    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. Gravity Model Dataset
# ---------------------------------------------------------------------------

class FlowDataset(Dataset):
    """
    PyTorch dataset for training the gravity model.

    Each sample corresponds to one origin tile. For each origin,
    we construct feature vectors for all candidate destinations
    and return the corresponding flow counts.
    """

    def __init__(
        self,
        origin_ids: List[str],
        tile2features: Dict[str, list],
        tile2centroid: Dict[str, tuple],
        o2d2flow: Dict[str, Dict[str, int]],
        all_tiles: List[str],
        dim_dests: int = 512,
        frac_true_dest: float = 0.5,
    ):
        """
        Args:
            origin_ids: List of origin tile IDs.
            tile2features: Mapping tile_id -> feature vector [population, poi_1, ..., poi_16].
            tile2centroid: Mapping tile_id -> (lat, lng).
            o2d2flow: Nested dict origin -> destination -> flow.
            all_tiles: List of all tile IDs.
            dim_dests: Maximum number of destinations to sample per origin.
            frac_true_dest: Fraction of sampled destinations that are true destinations.
        """
        self.origin_ids = origin_ids
        self.tile2features = tile2features
        self.tile2centroid = tile2centroid
        self.o2d2flow = o2d2flow
        self.all_tiles = all_tiles
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest

    def __len__(self) -> int:
        return len(self.origin_ids)

    def _sample_destinations(self, origin: str) -> List[str]:
        """Sample a mix of true and random destinations for an origin."""
        true_dests = list(self.o2d2flow.get(origin, {}).keys())
        n_total = min(self.dim_dests, len(self.all_tiles))
        n_true = min(int(n_total * self.frac_true_dest), len(true_dests))
        n_random = n_total - n_true

        sampled_true = list(np.random.choice(true_dests, size=n_true, replace=False)) if n_true > 0 else []
        remaining = list(set(self.all_tiles) - set(sampled_true))
        sampled_random = list(np.random.choice(remaining, size=min(n_random, len(remaining)), replace=False))

        dests = sampled_true + sampled_random
        np.random.shuffle(dests)
        return dests

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        origin = self.origin_ids[idx]
        dests = self._sample_destinations(origin)

        features = []
        flows = []
        origin_feat = self.tile2features[origin]
        origin_centroid = self.tile2centroid[origin]

        for dest in dests:
            dest_feat = self.tile2features[dest]
            dist = haversine_distance(origin_centroid, self.tile2centroid[dest])
            feat = origin_feat + dest_feat + [dist]
            features.append(feat)

            flow = self.o2d2flow.get(origin, {}).get(dest, 0)
            flows.append(flow)

        X = torch.tensor(features, dtype=torch.float32)
        T = torch.tensor(flows, dtype=torch.float32)
        return X, T


# ---------------------------------------------------------------------------
# 3. Temporal Dataset (for TimesFM Fine-Tuning)
# ---------------------------------------------------------------------------

class TemporalFlowDataset(Dataset):
    """
    Dataset for TimesFM fine-tuning.

    Each sample is a (context_window, covariates, target) tuple for
    a specific OD pair (i, j) at a reference time t.

    - context_window: historical flow values [T_{t-L}, ..., T_t] of shape (L,)
    - covariates: [a_ij, precipitation, temperature, hour] aligned with window, shape (L, 4)
    - target: T_{t+1} (scalar, one-step-ahead prediction)
    """

    def __init__(
        self,
        od_time_series: Dict[Tuple[str, str], np.ndarray],
        attractiveness: Dict[Tuple[str, str], np.ndarray],
        weather_data: Dict[str, np.ndarray],
        timestamps: np.ndarray,
        input_length: int = 32,
        horizon: int = 1,
        exclude_self_flows: bool = True,
    ):
        """
        Args:
            od_time_series: Mapping (origin, dest) -> time series array of flows.
            attractiveness: Mapping (origin, dest) -> time-varying attractiveness array.
            weather_data: Dict with 'precipitation_mm', 'temperature_c', 'hour_of_day' arrays.
            timestamps: Array of timestamps.
            input_length: Lookback window L (default 32).
            horizon: Forecast horizon H (default 1).
            exclude_self_flows: If True, skip OD pairs where origin == destination.
        """
        self.input_length = input_length
        self.horizon = horizon
        self.samples = []

        # Build sample index: (od_pair, start_time_idx)
        for (o, d), ts in od_time_series.items():
            if exclude_self_flows and o == d:
                continue
            T = len(ts)
            for t in range(input_length, T - horizon):
                self.samples.append((o, d, t))

        self.od_time_series = od_time_series
        self.attractiveness = attractiveness
        self.weather_data = weather_data
        self.timestamps = timestamps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, d, t = self.samples[idx]
        L = self.input_length
        H = self.horizon

        # Historical flow values: shape (L,)
        flow_ts = self.od_time_series[(o, d)]
        context = flow_ts[t - L : t].astype(np.float32)

        # Target: next H steps
        target = flow_ts[t : t + H].astype(np.float32)

        # Covariates: shape (L, num_covariates)
        attr_ts = self.attractiveness.get((o, d), np.zeros(len(flow_ts)))
        attr_window = attr_ts[t - L : t]

        precip = self.weather_data["precipitation_mm"][t - L : t]
        temp = self.weather_data["temperature_c"][t - L : t]
        hour = self.weather_data["hour_of_day"][t - L : t]

        covariates = np.stack([attr_window, precip, temp, hour], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(context),
            torch.from_numpy(covariates),
            torch.from_numpy(target),
        )


def create_gravity_dataloaders(
    data: dict,
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders for the gravity model.

    Args:
        data: Loaded data dictionary from load_od_tensor().
        config: Configuration dictionary.

    Returns:
        (train_loader, test_loader) tuple.
    """
    flows_df = data["flows_df"]
    tile2features = data["tile2features"]
    tile2centroid = data["tile2centroid"]

    # Apply log transform to population (first feature)
    for tid in tile2features:
        pop = max(tile2features[tid][0], 1e-6)
        tile2features[tid][0] = np.log(pop)

    # Build flow dictionary (aggregate over time for gravity training)
    agg_flows = flows_df.groupby(["origin", "destination"])["flow"].sum().reset_index()
    o2d2flow = build_o2d2flow(agg_flows)

    # All tiles
    all_tiles = list(tile2features.keys())

    # Spatial train/test split (50/50 random split of origins)
    np.random.shuffle(all_tiles)
    split = len(all_tiles) // 2
    train_origins = all_tiles[:split]
    test_origins = all_tiles[split:]

    gravity_cfg = config.get("gravity", {})

    train_dataset = FlowDataset(
        origin_ids=train_origins,
        tile2features=tile2features,
        tile2centroid=tile2centroid,
        o2d2flow=o2d2flow,
        all_tiles=all_tiles,
        dim_dests=gravity_cfg.get("dim_dests", 512),
        frac_true_dest=gravity_cfg.get("frac_true_dest", 0.5),
    )

    test_dataset = FlowDataset(
        origin_ids=test_origins,
        tile2features=tile2features,
        tile2centroid=tile2centroid,
        o2d2flow=o2d2flow,
        all_tiles=all_tiles,
        dim_dests=int(1e9),  # Use all destinations for testing
        frac_true_dest=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=gravity_cfg.get("batch_size", 1),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    return train_loader, test_loader
