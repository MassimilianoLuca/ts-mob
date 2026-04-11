"""
Step 2: Train the Gravity-Inspired Geographic & Social-Aware Module

Trains the Deep Gravity-inspired feed-forward network for destination
attractiveness estimation using tile features (population + POIs)
and pairwise distances.

From the paper (Section 4.2):
  - 7 fully-connected layers with LeakyReLU + Dropout
  - Cross-entropy loss (multinomial classification)
  - RMSprop optimizer, lr=5e-6, momentum=0.9
  - 15 epochs

Usage:
    python scripts/02_train_gravity.py --config configs/default.yaml --data_dir data/synthetic
"""

import argparse
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, set_seed, get_device, save_checkpoint
from src.gravity_model import GravityModel, train_gravity_epoch, evaluate_gravity
from src.data_loader import FlowDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Step 2: Train gravity model")
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
    print("  Step 2: Train Gravity Model")
    print("=" * 60)
    print(f"  Device: {device}")

    # Load preprocessed data
    processed_path = os.path.join(data_dir, "processed", "preprocessed.pkl")
    with open(processed_path, "rb") as f:
        preprocessed = pickle.load(f)

    tile_ids = preprocessed["tile_ids"]
    tile2features = preprocessed["tile2features"]
    tile2centroid = preprocessed["tile2centroid"]
    o2d2flow = preprocessed["o2d2flow_train"]

    # Apply log transform to population
    for tid in tile2features:
        pop = max(tile2features[tid][0], 1e-6)
        tile2features[tid][0] = np.log(pop)

    # Train/Test split (50/50 spatial split for gravity model)
    all_tiles = list(tile_ids)
    np.random.shuffle(all_tiles)
    split = len(all_tiles) // 2
    train_origins = all_tiles[:split]
    test_origins = all_tiles[split:]

    gravity_cfg = config.get("gravity", {})

    # Create datasets
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
        dim_dests=min(len(all_tiles), 1000),
        frac_true_dest=0.0,
    )

    train_loader = DataLoader(train_dataset, batch_size=gravity_cfg.get("batch_size", 1), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Compute input dimension
    sample_feat = tile2features[all_tiles[0]]
    dim_input = len(sample_feat) * 2 + 1  # origin_feat + dest_feat + distance
    print(f"  Input dimension: {dim_input}")
    print(f"  Train origins: {len(train_origins)}, Test origins: {len(test_origins)}")

    # Initialize model
    model = GravityModel(
        dim_input=dim_input,
        dim_hidden=gravity_cfg.get("dim_hidden", 256),
        dropout_p=gravity_cfg.get("dropout", 0.35),
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer: RMSprop as specified in the paper
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=gravity_cfg.get("lr", 5e-6),
        momentum=gravity_cfg.get("momentum", 0.9),
    )

    # Training loop
    epochs = gravity_cfg.get("epochs", 15)
    print(f"\n  Training for {epochs} epochs...\n")

    best_cpc = 0.0
    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "gravity_model.pt")

    # Initial evaluation
    cpc = evaluate_gravity(model, test_loader, device)
    print(f"  Epoch  0 (init) | CPC: {cpc:.4f}")

    for epoch in range(1, epochs + 1):
        # Set new random seeds per epoch
        set_seed(config.get("seed", 1234) + epoch)

        loss = train_gravity_epoch(model, optimizer, train_loader, device)
        cpc = evaluate_gravity(model, test_loader, device)

        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss:.6f} | CPC: {cpc:.4f}")

        if cpc > best_cpc:
            best_cpc = cpc
            save_checkpoint(model, optimizer, epoch, model_path)

    print(f"\n  Best CPC: {best_cpc:.4f}")
    print(f"  Model saved to: {model_path}")
    print("\n  Step 2 complete!")


if __name__ == "__main__":
    main()
