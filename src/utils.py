"""
Utility functions for TS-Mob.

Includes:
- Haversine distance computation
- Configuration loading
- Reproducibility setup
- Weather data fetching from Open-Meteo API
"""

import os
import random
import yaml
import numpy as np
import torch
from math import sqrt, sin, cos, pi, asin


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """Get the torch device from config."""
    device_str = config.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def haversine_distance(lat_lng1: tuple, lat_lng2: tuple) -> float:
    """
    Compute the great-circle distance between two points on Earth
    using the Haversine formula.

    Args:
        lat_lng1: (latitude, longitude) of point 1 in degrees
        lat_lng2: (latitude, longitude) of point 2 in degrees

    Returns:
        Distance in kilometers.
    """
    lat1, lng1 = [l * pi / 180 for l in lat_lng1]
    lat2, lng2 = [l * pi / 180 for l in lat_lng2]
    dlat, dlng = lat1 - lat2, lng1 - lng2
    ds = 2 * asin(sqrt(sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2.0) ** 2))
    return 6371.01 * ds


def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly: bool = True,
) -> dict:
    """
    Fetch historical weather data from Open-Meteo API.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        hourly: If True, fetch hourly data; otherwise daily.

    Returns:
        Dictionary with keys 'time', 'precipitation_mm', 'temperature_c'.
    """
    import requests

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,apparent_temperature" if hourly else None,
        "daily": "precipitation_sum,apparent_temperature_max" if not hourly else None,
        "timezone": "auto",
    }
    params = {k: v for k, v in params.items() if v is not None}

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    if hourly:
        hourly_data = data.get("hourly", {})
        return {
            "time": hourly_data.get("time", []),
            "precipitation_mm": hourly_data.get("precipitation", []),
            "temperature_c": hourly_data.get("apparent_temperature", []),
        }
    else:
        daily_data = data.get("daily", {})
        return {
            "time": daily_data.get("time", []),
            "precipitation_mm": daily_data.get("precipitation_sum", []),
            "temperature_c": daily_data.get("apparent_temperature_max", []),
        }


def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch
