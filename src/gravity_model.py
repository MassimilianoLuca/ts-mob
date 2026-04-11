"""
Geographic- and Social-Aware Module (Gravity-Inspired Model).

This module implements the Deep Gravity-inspired feed-forward neural network
that computes destination-attractiveness profiles from open geographic and
social signals (Section 4.2 of the paper).

Architecture (7 fully-connected layers as described in the paper):
  - Input: concat[x_i, x_j, r_ij]
    where x_i = [population_i, poi_cat_1, ..., poi_cat_16] (17 dims)
          x_j = [population_j, poi_cat_1, ..., poi_cat_16] (17 dims)
          r_ij = Haversine distance between centroids     (1 dim)
    Total input dimension = 35

  - 7 hidden layers with LeakyReLU + Dropout
  - Linear output layer -> scalar score per OD pair
  - Softmax across all destinations -> attractiveness distribution

Training:
  - Cross-entropy loss (multinomial classification of destination choice)
  - RMSprop optimizer, lr=5e-6, momentum=0.9
"""

import numpy as np
import torch
import torch.nn as nn


class GravityModel(nn.Module):
    """
    Deep Gravity-inspired feed-forward network.

    For a given origin i, processes all candidate destinations {j}
    in parallel and outputs a scalar score per OD pair. Scores are
    converted to destination-attractiveness via softmax.

    Architecture: 7 hidden layers as described in Section 4.2.
    """

    def __init__(self, dim_input: int = 35, dim_hidden: int = 256, dropout_p: float = 0.35):
        """
        Args:
            dim_input: Input feature dimension (default 35 = 17 + 17 + 1).
            dim_hidden: Hidden layer dimension.
            dropout_p: Dropout probability.
        """
        super().__init__()

        layers = []

        # Layer 1: input -> hidden
        layers.extend([
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
        ])

        # Layers 2-5: hidden -> hidden
        for _ in range(4):
            layers.extend([
                nn.Linear(dim_hidden, dim_hidden),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
            ])

        # Layer 6: hidden -> hidden // 2
        layers.extend([
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
        ])

        # Layer 7: hidden // 2 -> hidden // 2
        layers.extend([
            nn.Linear(dim_hidden // 2, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
        ])

        self.hidden = nn.Sequential(*layers)

        # Output: scalar score per OD pair
        self.output_layer = nn.Linear(dim_hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_destinations, dim_input)
               or (n_destinations, dim_input) for a single origin.

        Returns:
            Scalar scores of shape (batch, n_destinations, 1)
            or (n_destinations, 1).
        """
        h = self.hidden(x)
        out = self.output_layer(h)
        return out

    def predict_attractiveness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute softmax attractiveness over destinations.

        Args:
            x: Input tensor of shape (batch, n_destinations, dim_input).

        Returns:
            Attractiveness distribution of shape (batch, n_destinations),
            summing to 1 across destinations for each origin.
        """
        scores = self.forward(x).squeeze(-1)  # (batch, n_destinations)
        return torch.softmax(scores, dim=-1)

    def loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss for destination choice (multinomial classification).

        H = -sum_i sum_j (y_{ij} / O_i) * ln(p_ij)

        Args:
            scores: Raw output scores, shape (batch, n_destinations, 1).
            targets: Flow counts, shape (batch, n_destinations).

        Returns:
            Scalar loss.
        """
        log_probs = torch.log_softmax(scores.squeeze(-1), dim=-1)
        return -(targets * log_probs).sum()


def build_od_features(
    origin_features: np.ndarray,
    dest_features: np.ndarray,
    distance: float,
) -> np.ndarray:
    """
    Construct the input feature vector for an OD pair.

    x(l_i, l_j) = concat[x_i, x_j, r_ij]

    where x_i/x_j = [log_population, poi_cat_1, ..., poi_cat_16]
    and r_ij = Haversine distance between centroids.

    Args:
        origin_features: Feature vector of origin tile (17,).
        dest_features: Feature vector of destination tile (17,).
        distance: Haversine distance between centroids (km).

    Returns:
        Concatenated feature vector (35,).
    """
    return np.concatenate([origin_features, dest_features, [distance]])


def train_gravity_epoch(
    model: GravityModel,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Train the gravity model for one epoch.

    Args:
        model: GravityModel instance.
        optimizer: Optimizer (RMSprop).
        train_loader: DataLoader yielding (features, flow_counts).
        device: Torch device.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        optimizer.zero_grad()
        scores = model(batch_x)
        loss = model.loss(scores, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_gravity(
    model: GravityModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the gravity model using CPC.

    Args:
        model: Trained GravityModel.
        test_loader: DataLoader yielding (features, flow_counts).
        device: Torch device.

    Returns:
        Average CPC across test origins.
    """
    from .metrics import cpc

    model.eval()
    total_cpc = 0.0
    n_origins = 0

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device).float()
        batch_y = batch_y.to(device).float()

        # Predict attractiveness probabilities
        probs = model.predict_attractiveness(batch_x)

        for i in range(batch_x.shape[0]):
            y_true = batch_y[i].cpu().numpy()
            total_flow = y_true.sum()
            if total_flow > 0:
                y_pred = probs[i].cpu().numpy() * total_flow
                total_cpc += cpc(y_true, y_pred)
                n_origins += 1

    return total_cpc / max(n_origins, 1)
