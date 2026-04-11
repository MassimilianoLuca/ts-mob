"""
TS-Mob Full Inference Pipeline.

At inference time, TS-Mob operates in two steps (Section 4.4):
  1. The geographic- and social-aware block computes OD-specific
     attractiveness values a_ij using open data.
  2. For each OD pair (i, j), the temporal block receives the
     context window of L=32 flow observations plus covariates
     and outputs the one-step-ahead prediction.

Output: predicted OD tensor T_hat_{t+1, i, j}
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from .gravity_model import GravityModel
from .attractiveness import compute_attractiveness
from .timesfm_finetune import forecast_with_timesfm
from .metrics import evaluate_all


def run_inference(
    gravity_model: GravityModel,
    timesfm_model,
    tile2features: Dict[str, list],
    tile2centroid: Dict[str, tuple],
    tile_ids: list,
    od_time_series: Dict[Tuple[str, str], np.ndarray],
    weather_data: Dict[str, np.ndarray],
    test_start_idx: int,
    input_length: int = 32,
    horizon: int = 1,
    device: torch.device = torch.device("cpu"),
    exclude_self: bool = True,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """
    Run the full TS-Mob inference pipeline.

    Step 1: Compute attractiveness a_ij from gravity model
    Step 2: Forecast each OD flow with fine-tuned TimesFM + covariates

    Args:
        gravity_model: Trained GravityModel.
        timesfm_model: Fine-tuned TimesFM model.
        tile2features: Mapping tile_id -> feature vector.
        tile2centroid: Mapping tile_id -> (lat, lng).
        tile_ids: List of all tile IDs.
        od_time_series: Mapping (origin, dest) -> flow time series.
        weather_data: Dict with precipitation_mm, temperature_c, hour_of_day arrays.
        test_start_idx: Index where the test period begins.
        input_length: Lookback window L.
        horizon: Forecast horizon H.
        device: Torch device.
        exclude_self: If True, skip self-flows.

    Returns:
        (predictions, ground_truth) dictionaries mapping OD pairs to arrays.
    """
    # Step 1: Compute attractiveness
    print("\n=== Step 1: Computing attractiveness (gravity block) ===")
    attractiveness = compute_attractiveness(
        gravity_model, tile2features, tile2centroid, tile_ids, device, exclude_self
    )

    # Step 2: Forecast with TimesFM
    print("\n=== Step 2: Forecasting with TimesFM ===")
    predictions = {}
    ground_truth = {}

    precip = np.array(weather_data["precipitation_mm"], dtype=np.float32)
    temp = np.array(weather_data["temperature_c"], dtype=np.float32)
    hour = np.array(weather_data["hour_of_day"], dtype=np.float32)

    od_pairs = [(o, d) for (o, d) in od_time_series.keys()
                if not (exclude_self and o == d)]

    for (o, d) in tqdm(od_pairs, desc="Forecasting OD pairs"):
        flow_ts = od_time_series[(o, d)]
        T = len(flow_ts)

        # Attractiveness value (static, repeated across time)
        attr_val = attractiveness.get((o, d), 0.0)

        preds = []
        truths = []

        for t in range(test_start_idx, T - horizon):
            if t < input_length:
                continue

            # Context window
            context = flow_ts[t - input_length : t].astype(np.float32)

            # Build covariate matrix for context + horizon
            total_len = input_length + horizon
            start = t - input_length
            end = t + horizon

            cov = np.stack([
                np.full(total_len, attr_val),
                precip[start:end],
                temp[start:end],
                hour[start:end],
            ], axis=-1).astype(np.float32)

            # Forecast
            pred = forecast_with_timesfm(
                timesfm_model, context, cov, horizon=horizon
            )

            preds.append(pred)
            truths.append(flow_ts[t : t + horizon].astype(np.float32))

        if preds:
            predictions[(o, d)] = np.array(preds)
            ground_truth[(o, d)] = np.array(truths)

    return predictions, ground_truth


def evaluate_predictions(
    predictions: Dict[Tuple[str, str], np.ndarray],
    ground_truth: Dict[Tuple[str, str], np.ndarray],
) -> dict:
    """
    Evaluate predictions aggregated across all OD pairs.

    Args:
        predictions: Dict mapping OD pairs to predicted flows.
        ground_truth: Dict mapping OD pairs to actual flows.

    Returns:
        Dictionary with RMSE, MAE, and CPC.
    """
    all_preds = []
    all_truths = []
    for key in predictions:
        all_preds.append(predictions[key].flatten())
        all_truths.append(ground_truth[key].flatten())

    all_preds = np.concatenate(all_preds)
    all_truths = np.concatenate(all_truths)

    results = evaluate_all(all_truths, all_preds)

    print("\n" + "=" * 50)
    print("  TS-Mob Evaluation Results")
    print("=" * 50)
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  CPC:  {results['cpc']:.4f}")
    print("=" * 50)

    return results
