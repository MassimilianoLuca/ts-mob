"""
Generate synthetic mobility data for testing the TS-Mob pipeline.

Creates a small, self-contained dataset with:
- n_tiles tiles on a grid with random POI features and populations
- Synthetic OD flows following a gravity-like distribution
- Synthetic weather data
- Proper time series structure for temporal training

Usage:
    python data/generate_synthetic.py --output_dir data/synthetic --n_tiles 10 --n_timesteps 100
"""

import argparse
import os

import numpy as np
import pandas as pd


# Overture top-level POI categories
POI_CATEGORIES = [
    "accommodation", "active_life", "arts_and_entertainment",
    "attractions_and_activities", "automotive", "beauty_and_spa",
    "eat_and_drink", "education", "financial_service",
    "health_and_medical", "professional_services",
    "public_service_and_government", "real_estate",
    "religious_organization", "retail", "travel",
]


def generate_synthetic_data(
    output_dir: str,
    n_tiles: int = 10,
    n_timesteps: int = 100,
    seed: int = 42,
) -> None:
    """
    Generate synthetic data matching the expected TS-Mob format.

    Args:
        output_dir: Directory to write output files.
        n_tiles: Number of spatial tiles.
        n_timesteps: Number of time steps.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    tile_ids = [f"tile_{i:03d}" for i in range(n_tiles)]

    # ----------------------------------------------------------------
    # 1. Tile centroids (random grid within a city-like bounding box)
    # ----------------------------------------------------------------
    grid_size = int(np.ceil(np.sqrt(n_tiles)))
    lats = np.linspace(40.70, 40.80, grid_size)   # NYC-like latitude
    lngs = np.linspace(-74.02, -73.95, grid_size)  # NYC-like longitude

    centroids = []
    for i, tid in enumerate(tile_ids):
        row, col = divmod(i, grid_size)
        lat = lats[min(row, len(lats) - 1)] + rng.uniform(-0.002, 0.002)
        lng = lngs[min(col, len(lngs) - 1)] + rng.uniform(-0.002, 0.002)
        centroids.append({"tile_id": tid, "latitude": lat, "longitude": lng})

    centroids_df = pd.DataFrame(centroids)
    centroids_df.to_csv(os.path.join(output_dir, "tile_centroids.csv"), index=False)
    print(f"  Generated tile_centroids.csv ({n_tiles} tiles)")

    # ----------------------------------------------------------------
    # 2. Tile features (population + 16 POI categories)
    # ----------------------------------------------------------------
    features = []
    for tid in tile_ids:
        pop = rng.integers(500, 50000)
        pois = {cat: int(rng.integers(0, 100)) for cat in POI_CATEGORIES}
        features.append({"tile_id": tid, "population": pop, **pois})

    features_df = pd.DataFrame(features)
    features_df.to_csv(os.path.join(output_dir, "tile_features.csv"), index=False)
    print(f"  Generated tile_features.csv ({n_tiles} tiles, {len(POI_CATEGORIES)} POI categories)")

    # ----------------------------------------------------------------
    # 3. OD flows (gravity-like synthetic flows over time)
    # ----------------------------------------------------------------
    # Compute distances for gravity-like flow generation
    centroid_dict = {row["tile_id"]: (row["latitude"], row["longitude"])
                     for _, row in centroids_df.iterrows()}
    pop_dict = {row["tile_id"]: row["population"]
                for _, row in features_df.iterrows()}

    flows = []
    for t in range(n_timesteps):
        # Add temporal variation (hourly pattern)
        hour = t % 24
        # Rush hour multiplier
        time_mult = 1.0 + 0.5 * np.exp(-((hour - 8) ** 2) / 8) + 0.3 * np.exp(-((hour - 18) ** 2) / 8)

        for o in tile_ids:
            for d in tile_ids:
                if o == d:
                    continue

                # Gravity-like flow: proportional to pop_o * pop_d / dist^2
                lat1, lng1 = centroid_dict[o]
                lat2, lng2 = centroid_dict[d]
                dist = np.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2) * 111  # Rough km

                pop_o = pop_dict[o]
                pop_d = pop_dict[d]

                # Base flow with distance decay
                base_flow = (pop_o * pop_d) / (max(dist, 0.1) ** 2 * 1e8)
                flow = int(max(0, rng.poisson(max(base_flow * time_mult, 0.01))))

                if flow > 0:
                    flows.append({
                        "timestamp": t,
                        "origin": o,
                        "destination": d,
                        "flow": flow,
                    })

    flows_df = pd.DataFrame(flows)
    flows_df.to_csv(os.path.join(output_dir, "od_flows.csv"), index=False)
    print(f"  Generated od_flows.csv ({len(flows_df)} flow records, {n_timesteps} timesteps)")

    # ----------------------------------------------------------------
    # 4. Weather data
    # ----------------------------------------------------------------
    weather = []
    for t in range(n_timesteps):
        hour = t % 24
        weather.append({
            "timestamp": t,
            "precipitation_mm": max(0, rng.normal(0.5, 1.5)),
            "temperature_c": 15 + 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2),
            "hour_of_day": hour,
        })

    weather_df = pd.DataFrame(weather)
    weather_df.to_csv(os.path.join(output_dir, "weather.csv"), index=False)
    print(f"  Generated weather.csv ({n_timesteps} rows)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\nSynthetic dataset generated in: {output_dir}")
    print(f"  Tiles: {n_tiles}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  OD pairs (excl. self): {n_tiles * (n_tiles - 1)}")
    print(f"  Total flow records: {len(flows_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic mobility data")
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                        help="Output directory for synthetic data")
    parser.add_argument("--n_tiles", type=int, default=10,
                        help="Number of spatial tiles")
    parser.add_argument("--n_timesteps", type=int, default=100,
                        help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    generate_synthetic_data(
        output_dir=args.output_dir,
        n_tiles=args.n_tiles,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
    )
