"""
Attractiveness computation module.

After the gravity model is trained, this module uses it to compute
destination-attractiveness scores a_ij for all OD pairs. These scores
serve as covariates for the temporal TimesFM block.

From the paper (Section 4.2):
    a_{ij,tau} = exp(s(l_i, l_j)) / sum_k exp(s(l_i, l_k))

where s(l_i, l_j) is the score output by the gravity model.
The attractiveness varies over time because the population (living
population) input changes across timestamps.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .gravity_model import GravityModel
from .utils import haversine_distance


@torch.no_grad()
def compute_attractiveness(
    model: GravityModel,
    tile2features: Dict[str, list],
    tile2centroid: Dict[str, tuple],
    tile_ids: list,
    device: torch.device,
    exclude_self: bool = True,
) -> Dict[Tuple[str, str], float]:
    """
    Compute static attractiveness scores for all OD pairs.

    For each origin i, computes the softmax attractiveness distribution
    over all possible destinations.

    Args:
        model: Trained GravityModel.
        tile2features: Mapping tile_id -> feature vector.
        tile2centroid: Mapping tile_id -> (lat, lng).
        tile_ids: List of all tile IDs.
        device: Torch device.
        exclude_self: If True, exclude self-loops (i == j).

    Returns:
        Dictionary mapping (origin, dest) -> attractiveness score.
    """
    model.eval()
    n = len(tile_ids)
    attractiveness = {}

    for i, origin in enumerate(tqdm(tile_ids, desc="Computing attractiveness")):
        origin_feat = tile2features[origin]
        origin_centroid = tile2centroid[origin]

        # Build feature vectors for all destinations
        features = []
        dest_list = []
        for dest in tile_ids:
            if exclude_self and origin == dest:
                continue
            dest_feat = tile2features[dest]
            dist = haversine_distance(origin_centroid, tile2centroid[dest])
            feat = origin_feat + dest_feat + [dist]
            features.append(feat)
            dest_list.append(dest)

        if len(features) == 0:
            continue

        # Forward pass
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_dest, dim_input)
        probs = model.predict_attractiveness(X).squeeze(0)  # (n_dest,)
        probs_np = probs.cpu().numpy()

        # Store attractiveness
        for j, dest in enumerate(dest_list):
            attractiveness[(origin, dest)] = float(probs_np[j])

    return attractiveness


def compute_time_varying_attractiveness(
    model: GravityModel,
    tile2features_by_time: Dict[int, Dict[str, list]],
    tile2centroid: Dict[str, tuple],
    tile_ids: list,
    timestamps: list,
    device: torch.device,
    exclude_self: bool = True,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute time-varying attractiveness when population changes over time.

    In the paper, living population (as opposed to registered population)
    changes over time, so attractiveness is recomputed per timestep.

    Args:
        model: Trained GravityModel.
        tile2features_by_time: Mapping timestamp -> tile_id -> feature vector.
        tile2centroid: Mapping tile_id -> (lat, lng).
        tile_ids: List of all tile IDs.
        timestamps: List of timestamps.
        device: Torch device.
        exclude_self: If True, exclude self-loops.

    Returns:
        Dictionary mapping (origin, dest) -> np.array of attractiveness over time.
    """
    model.eval()

    # Initialize storage
    od_attractiveness = {}
    for o in tile_ids:
        for d in tile_ids:
            if exclude_self and o == d:
                continue
            od_attractiveness[(o, d)] = np.zeros(len(timestamps))

    for t_idx, ts in enumerate(tqdm(timestamps, desc="Time-varying attractiveness")):
        tile2features = tile2features_by_time.get(ts, tile2features_by_time.get(timestamps[0]))

        # Compute attractiveness at this timestep
        attr_t = compute_attractiveness(
            model, tile2features, tile2centroid, tile_ids, device, exclude_self
        )

        for (o, d), val in attr_t.items():
            od_attractiveness[(o, d)][t_idx] = val

    return od_attractiveness


def save_attractiveness(attractiveness: dict, output_path: str) -> None:
    """Save attractiveness to disk as numpy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, attractiveness)
    print(f"Attractiveness saved to {output_path} ({len(attractiveness)} OD pairs)")


def load_attractiveness(path: str) -> dict:
    """Load attractiveness from disk."""
    data = np.load(path, allow_pickle=True).item()
    print(f"Attractiveness loaded from {path} ({len(data)} OD pairs)")
    return data
