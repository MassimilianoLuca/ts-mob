"""
TimesFM Fine-Tuning Module.

Fine-tunes the pre-trained TimesFM 2.5 model for OD flow prediction,
conditioned on gravity-based attractiveness covariates and exogenous
signals (weather, calendar).

From the paper (Section 4.3 & Appendix B.3):
  - Optimizer: SGD with momentum 0.9
  - Learning rate: 5e-4 with linear warmup (25 epochs) + cosine decay
  - Total epochs: 50
  - Gradient clipping: 1.0
  - Input length: L = 32
  - Loss: MSE
  - Covariates: [attractiveness, precipitation_mm, temperature_c, hour_of_day]
"""

import os
import math
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_lr_schedule(
    epoch: int,
    warmup_epochs: int = 25,
    total_epochs: int = 50,
    base_lr: float = 5e-4,
) -> float:
    """
    Linear warmup + cosine decay learning rate schedule.

    Args:
        epoch: Current epoch (0-indexed).
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of epochs.
        base_lr: Base learning rate after warmup.

    Returns:
        Learning rate for the current epoch.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def load_timesfm_model(model_name: str = "google/timesfm-2.5-200m-pytorch"):
    """
    Load pre-trained TimesFM model.

    Requires: pip install timesfm[xreg]

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        TimesFM model instance.
    """
    import timesfm

    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    print(f"Loaded TimesFM model: {model_name}")
    return model


def prepare_temporal_data(
    od_time_series: Dict[Tuple[str, str], np.ndarray],
    attractiveness: Dict[Tuple[str, str], np.ndarray],
    weather_data: Dict[str, np.ndarray],
    input_length: int = 32,
    horizon: int = 1,
    exclude_self: bool = True,
) -> Tuple[list, list, list]:
    """
    Prepare data for TimesFM fine-tuning.

    Constructs sliding windows of (context, covariates, target)
    for each OD pair.

    Args:
        od_time_series: Mapping (origin, dest) -> flow time series.
        attractiveness: Mapping (origin, dest) -> attractiveness time series.
        weather_data: Dict with precipitation_mm, temperature_c, hour_of_day arrays.
        input_length: Lookback window L.
        horizon: Forecast horizon H.
        exclude_self: Exclude self-flows.

    Returns:
        (contexts, covariates, targets) lists of numpy arrays.
    """
    contexts = []
    all_covariates = []
    targets = []

    precip = np.array(weather_data["precipitation_mm"], dtype=np.float32)
    temp = np.array(weather_data["temperature_c"], dtype=np.float32)
    hour = np.array(weather_data["hour_of_day"], dtype=np.float32)

    for (o, d), flow_ts in od_time_series.items():
        if exclude_self and o == d:
            continue

        T = len(flow_ts)
        attr_ts = attractiveness.get(
            (o, d),
            np.zeros(T, dtype=np.float32),
        )

        for t in range(input_length, T - horizon):
            # Context window
            ctx = flow_ts[t - input_length : t].astype(np.float32)

            # Covariates aligned with context window
            cov = np.stack([
                attr_ts[t - input_length : t],
                precip[t - input_length : t],
                temp[t - input_length : t],
                hour[t - input_length : t],
            ], axis=-1).astype(np.float32)

            # Target
            tgt = flow_ts[t : t + horizon].astype(np.float32)

            contexts.append(ctx)
            all_covariates.append(cov)
            targets.append(tgt)

    return contexts, all_covariates, targets


def finetune_timesfm(
    model,
    contexts: list,
    covariates: list,
    targets: list,
    config: dict,
    device: torch.device,
    save_path: Optional[str] = None,
) -> dict:
    """
    Fine-tune TimesFM with covariates using the paper's training protocol.

    This function implements the fine-tuning loop described in Section 4.3
    and Appendix B.3 of the paper: SGD with linear warmup + cosine decay,
    MSE loss, gradient clipping.

    Note: This is a simplified fine-tuning loop. The actual TimesFM library
    may provide its own fine-tuning utilities. This implementation shows
    the conceptual flow matching the paper.

    Args:
        model: TimesFM model (or a wrapper supporting .forward()).
        contexts: List of context arrays, each of shape (L,).
        covariates: List of covariate arrays, each of shape (L, 4).
        targets: List of target arrays, each of shape (H,).
        config: TimesFM config section from YAML.
        device: Torch device.
        save_path: Optional path to save fine-tuned model.

    Returns:
        Dictionary with training history.
    """
    lr = config.get("lr", 5e-4)
    momentum = config.get("momentum", 0.9)
    gradient_clip = config.get("gradient_clip", 1.0)
    total_epochs = config.get("epochs", 50)
    warmup_epochs = config.get("warmup_epochs", 25)

    # Convert to tensors
    ctx_tensor = torch.stack([torch.from_numpy(c) for c in contexts]).to(device)
    cov_tensor = torch.stack([torch.from_numpy(c) for c in covariates]).to(device)
    tgt_tensor = torch.stack([torch.from_numpy(t) for t in targets]).to(device)

    n_samples = len(contexts)
    batch_size = min(64, n_samples)

    # Create a simple DataLoader
    dataset = torch.utils.data.TensorDataset(ctx_tensor, cov_tensor, tgt_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer: SGD with momentum as specified in the paper
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)

    # Loss: MSE as specified in the paper
    criterion = nn.MSELoss()

    history = {"epoch": [], "loss": [], "lr": []}

    print(f"\nFine-tuning TimesFM: {total_epochs} epochs, lr={lr}, "
          f"warmup={warmup_epochs}, {n_samples} samples")

    for epoch in range(total_epochs):
        # Update learning rate: linear warmup + cosine decay
        current_lr = get_lr_schedule(epoch, warmup_epochs, total_epochs, lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_ctx, batch_cov, batch_tgt in dataloader:
            optimizer.zero_grad()

            # Forward pass: the model predicts future flows from context + covariates
            # NOTE: The actual TimesFM API uses forecast() / forecast_with_covariates().
            #       This simplified version shows the conceptual training loop.
            #       In practice, you would use TimesFM's native fine-tuning support.
            predictions = model(batch_ctx, batch_cov)

            loss = criterion(predictions, batch_tgt)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_params, gradient_clip)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{total_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    # Save fine-tuned model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nFine-tuned model saved to {save_path}")

    return history


def forecast_with_timesfm(
    model,
    context: np.ndarray,
    covariates: np.ndarray,
    horizon: int = 1,
) -> np.ndarray:
    """
    Generate forecasts using the fine-tuned TimesFM model.

    This wraps the TimesFM forecast API with covariate support.

    Args:
        model: Fine-tuned TimesFM model.
        context: Historical flow values, shape (L,).
        covariates: Covariate matrix, shape (L + horizon, num_covariates).
        horizon: Number of steps to forecast.

    Returns:
        Predicted flows, shape (horizon,).
    """
    import timesfm

    model.compile(
        timesfm.ForecastConfig(
            max_context=len(context),
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
        )
    )

    # Use the covariate forecast API
    # Covariates: attractiveness, precipitation, temperature, hour
    point_forecast, _ = model.forecast_with_covariates(
        inputs=[context],
        dynamic_numerical_covariates={
            "attractiveness": [covariates[:, 0]],
            "precipitation_mm": [covariates[:, 1]],
            "temperature_c": [covariates[:, 2]],
        },
        dynamic_categorical_covariates={
            "hour_of_day": [covariates[:, 3].astype(int)],
        },
        xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True,
    )

    return point_forecast[0, :horizon]
