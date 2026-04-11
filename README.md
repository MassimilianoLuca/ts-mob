# TS-Mob: Social and Geographical-Aware Time Series Foundation-Model Framework for Human Mobility Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TimesFM](https://img.shields.io/badge/TimesFM-2.5-green.svg)](https://github.com/google-research/timesfm)

## Paper Summary

TS-Mob is a two-component framework for human mobility prediction that combines:

1. **Geographic & Social-Aware Module** — A gravity-inspired deep neural network that estimates, for each origin, an interpretable destination-attractiveness profile from open geographic and social signals (living population, Overture Maps POIs, and pairwise distances).

2. **Temporal Module** — A fine-tuned Time Series Foundation Model ([TimesFM 2.5](https://github.com/google-research/timesfm)) that predicts future OD flows conditioned on the attractiveness covariates, historical mobility volumes, and exogenous temporal signals (weather, calendar).

Instead of learning the joint dynamics of the entire OD tensor (as in traditional spatio-temporal models), TS-Mob decomposes the problem into **independent univariate time-series forecasting tasks** — one per origin-destination pair — injecting structural geographic/social knowledge as covariates. This design enables geographically-aware predictions without training a fully spatio-temporal model from scratch.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TS-Mob Pipeline                             │
│                                                                     │
│  ┌──────────────────────────────────┐                               │
│  │ Geographic & Social-Aware Module │                               │
│  │  (Deep Gravity-Inspired)        │                               │
│  │                                  │                               │
│  │  Input: x(l_i, l_j) = concat[   │                               │
│  │    population_i (living pop),    │    ┌─────────────────────┐    │
│  │    POI_density_i (16 cats),      │───▶│   Attractiveness    │    │
│  │    population_j,                 │    │   a_ij (softmax)    │    │
│  │    POI_density_j,                │    └────────┬────────────┘    │
│  │    Haversine_distance(i,j)       │             │                 │
│  │  ]                               │             ▼                 │
│  │                                  │    ┌─────────────────────┐    │
│  │  Architecture: 7 FF layers       │    │  Temporal Module     │    │
│  │  + LeakyReLU + Dropout          │    │  (Fine-tuned         │    │
│  │  + Softmax over destinations    │    │   TimesFM 2.5)       │    │
│  └──────────────────────────────────┘    │                     │    │
│                                          │  Covariates:        │    │
│  Historical Flows ──────────────────────▶│  - a_ij             │    │
│  [T_{t-L}, ..., T_t]                    │  - precipitation    │    │
│  Weather (precip, temp) ────────────────▶│  - temperature      │───▶ T̂_{t+1,i,j}
│  Calendar (hour) ───────────────────────▶│  - hour_of_day      │    │
│                                          └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Results

TS-Mob consistently outperforms both traditional methods (MA, ARIMA, VAR) and deep learning baselines (ST-ResNet, DMVSTNet, ACMF, CrowdNet, DSTAGNN) across three mobility datasets:

| Dataset | Metric | Best Baseline | TS-Mob |
|---------|--------|--------------|--------|
| Bike NYC | RMSE | (varies) | **Best** |
| Taxi Beijing | RMSE | (varies) | **Best** |
| Spain OD | RMSE | (varies) | **Best** |

---

## Repository Structure

```
TS-Mob-Model/
├── README.md                         # This file
├── requirements.txt                  # Dependencies
├── configs/
│   └── default.yaml                  # Default hyperparameters (from paper)
├── data/
│   ├── README.md                     # Dataset format & download instructions
│   └── generate_synthetic.py         # Generate synthetic test data
├── src/
│   ├── __init__.py
│   ├── utils.py                      # Haversine distance, config, weather API
│   ├── metrics.py                    # RMSE, MAE, CPC evaluation metrics
│   ├── gravity_model.py              # Geographic & Social-Aware Module
│   ├── data_loader.py                # Data loading + dataset classes
│   ├── attractiveness.py             # Compute attractiveness covariates
│   ├── timesfm_finetune.py           # TimesFM fine-tuning with covariates
│   └── inference.py                  # Full TS-Mob inference pipeline
├── scripts/
│   ├── 01_prepare_data.py            # Step 1: Preprocess raw data
│   ├── 02_train_gravity.py           # Step 2: Train gravity module
│   ├── 03_compute_attractiveness.py  # Step 3: Compute a_ij covariates
│   ├── 04_finetune_timesfm.py        # Step 4: Fine-tune TimesFM
│   └── 05_evaluate.py               # Step 5: Run evaluation
└── results/                          # Output: models & metrics
```

---

## Installation

### Prerequisites
- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA support recommended)
- GPU with ≥ 12GB VRAM (recommended for TimesFM fine-tuning)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd TS-Mob-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install TimesFM

TimesFM must be installed separately with covariate support:

```bash
pip install timesfm[xreg]
```

For GPU acceleration, install the appropriate PyTorch version first:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install PyTorch (GPU)

```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start (Synthetic Data)

To test the pipeline without downloading real datasets:

```bash
# 1. Generate synthetic data
python data/generate_synthetic.py --output_dir data/synthetic --n_tiles 10 --n_timesteps 200

# 2. Prepare data
python scripts/01_prepare_data.py --data_dir data/synthetic

# 3. Train gravity model
python scripts/02_train_gravity.py --data_dir data/synthetic

# 4. Compute attractiveness covariates
python scripts/03_compute_attractiveness.py --data_dir data/synthetic

# 5. Fine-tune TimesFM (requires timesfm installed)
python scripts/04_finetune_timesfm.py --data_dir data/synthetic

# 6. Evaluate
python scripts/05_evaluate.py --data_dir data/synthetic
```

> **Note:** Steps 1–3 and 5–6 run without TimesFM installed. Step 4 (and the TimesFM path in Step 5) requires `pip install timesfm[xreg]`.

---

## Running with Real Datasets

### 1. Prepare Your Data

Place your data files in `data/<dataset_name>/` following the format described in [`data/README.md`](data/README.md):

```
data/bike_nyc/
├── od_flows.csv
├── tile_features.csv
├── tile_centroids.csv
└── weather.csv
```

### 2. Configure

Edit `configs/default.yaml` or create a custom config:

```yaml
data:
  dataset: "bike_nyc"
  data_dir: "./data"
  input_length: 32
  train_fraction: 0.8
```

### 3. Run the Pipeline

```bash
# Full pipeline
python scripts/01_prepare_data.py --config configs/default.yaml
python scripts/02_train_gravity.py --config configs/default.yaml
python scripts/03_compute_attractiveness.py --config configs/default.yaml
python scripts/04_finetune_timesfm.py --config configs/default.yaml
python scripts/05_evaluate.py --config configs/default.yaml
```

---

## Datasets

The paper evaluates TS-Mob on three mobility datasets:

| Dataset | Location | Period | Grid | Temporal Res. | Source |
|---------|----------|--------|------|---------------|--------|
| **BikeNYC** | New York | Apr–Sep 2014 | 16×8 | Hourly | [CitiBike](https://ride.citibikenyc.com/system-data) |
| **TaxiBJ** | Beijing | Feb 2008 (1 week) | 32×32 | 30 min | [T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) |
| **Spain OD** | Spain | Feb–Apr 2023 | 2,735 tiles | Hourly | [pySpainMobility](https://github.com/) |

**POI Data:** Points of Interest are obtained from [Overture Maps](https://overturemaps.org/) using the DuckDB interface (16 top-level categories).

**Weather Data:** Historical weather (precipitation, temperature) from the [Open-Meteo API](https://open-meteo.com/). A utility function is provided in `src/utils.py`:

```python
from src.utils import fetch_weather_data
weather = fetch_weather_data(
    latitude=40.75, longitude=-73.98,
    start_date="2014-04-01", end_date="2014-09-30"
)
```

---

## Hyperparameters

All hyperparameters are defined in `configs/default.yaml` and match the paper's Appendix B:

### Gravity Module
| Parameter | Value |
|-----------|-------|
| Hidden layers | 7 (FF + LeakyReLU + Dropout) |
| Hidden dim | 256 |
| Dropout | 0.35 |
| Optimizer | RMSprop |
| Learning rate | 5×10⁻⁶ |
| Momentum | 0.9 |
| Epochs | 15 |
| Input dimension | 35 (17 origin + 17 dest + 1 distance) |

### TimesFM Fine-Tuning
| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 5×10⁻⁴ |
| Momentum | 0.9 |
| Gradient clip | 1.0 |
| Total epochs | 50 |
| Warmup epochs | 25 (linear) |
| LR schedule | Cosine decay (after warmup) |
| Input length (L) | 32 |

### Evaluation Metrics
- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE** — Mean Absolute Error (lower is better)
- **CPC** — Common Part of Commuters (higher is better, range [0, 1])

---

## Computational Requirements

From the paper (Appendix B.4):
- **Server:** 126 GB RAM
- **GPU:** 2× NVIDIA GeForce RTX 2080 Ti (12 GB each)

For testing with synthetic data, a CPU-only machine is sufficient.

---

## Citation

```bibtex
@inproceedings{tsmob2025,
  title={TS-Mob: Social and Geographical-Aware Time Series Foundation-Model Framework for Human Mobility Prediction},
  author={YYY et al.},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

---

## License

This project is provided for research purposes. See corresponding licenses for [TimesFM](https://github.com/google-research/timesfm/blob/master/LICENSE) and [Deep Gravity](https://github.com/scikit-mobility/DeepGravity).
