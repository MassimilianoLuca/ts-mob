"""
Step 3: Compute Attractiveness Covariates

Uses the trained gravity model to compute destination-attractiveness
scores a_ij for all OD pairs. These scores serve as covariates for
the temporal TimesFM module.

From the paper (Section 4.2):
    a_{ij} = softmax(s(l_i, l_j)) over all destinations j

Usage:
    python scripts/03_compute_attractiveness.py --config configs/default.yaml --data_dir data/synthetic
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, set_seed, get_device, load_checkpoint
from src.gravity_model import GravityModel
from src.attractiveness import compute_attractiveness, save_attractiveness


def main():
    parser = argparse.ArgumentParser(description="Step 3: Compute attractiveness")
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

    print("=" * 60)
    print("  Step 3: Compute Attractiveness Covariates")
    print("=" * 60)

    # Load preprocessed data
    processed_path = os.path.join(data_dir, "processed", "preprocessed.pkl")
    with open(processed_path, "rb") as f:
        preprocessed = pickle.load(f)

    tile_ids = preprocessed["tile_ids"]
    tile2features = preprocessed["tile2features"]
    tile2centroid = preprocessed["tile2centroid"]

    # Apply log transform to population
    for tid in tile2features:
        pop = max(tile2features[tid][0], 1e-6)
        tile2features[tid][0] = np.log(pop)

    # Compute input dimension
    sample_feat = tile2features[tile_ids[0]]
    dim_input = len(sample_feat) * 2 + 1

    # Load trained gravity model
    gravity_cfg = config.get("gravity", {})
    results_dir = config.get("output", {}).get("results_dir", "results")
    model_path = os.path.join(results_dir, "gravity_model.pt")

    model = GravityModel(
        dim_input=dim_input,
        dim_hidden=gravity_cfg.get("dim_hidden", 256),
        dropout_p=gravity_cfg.get("dropout", 0.35),
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=gravity_cfg.get("lr", 5e-6))
    load_checkpoint(model, optimizer, model_path, device)

    # Compute attractiveness for all OD pairs
    print(f"\n  Computing attractiveness for {len(tile_ids)} tiles...")
    attractiveness = compute_attractiveness(
        model=model,
        tile2features=tile2features,
        tile2centroid=tile2centroid,
        tile_ids=tile_ids,
        device=device,
        exclude_self=config["data"].get("exclude_self_flows", True),
    )

    # Save attractiveness
    attr_path = os.path.join(data_dir, "attractiveness.npy")
    save_attractiveness(attractiveness, attr_path)

    # Print statistics
    attr_values = list(attractiveness.values())
    print(f"\n  Attractiveness statistics:")
    print(f"    OD pairs:  {len(attr_values)}")
    print(f"    Mean:      {np.mean(attr_values):.6f}")
    print(f"    Std:       {np.std(attr_values):.6f}")
    print(f"    Min:       {np.min(attr_values):.6f}")
    print(f"    Max:       {np.max(attr_values):.6f}")

    # Also save as part of preprocessed data
    preprocessed["attractiveness"] = attractiveness
    with open(processed_path, "wb") as f:
        pickle.dump(preprocessed, f)

    print(f"\n  Updated preprocessed data with attractiveness")
    print("\n  Step 3 complete!")


if __name__ == "__main__":
    main()
