# Data Directory

This directory should contain the mobility datasets. The code expects the following structure for each dataset:

## Expected File Structure

```
data/
├── <dataset_name>/
│   ├── od_flows.csv             # Origin-destination flows
│   ├── tile_features.csv        # Tile features (population + POIs)
│   ├── tile_centroids.csv       # Tile centroid coordinates
│   └── weather.csv              # Weather data (precipitation, temperature)
├── generate_synthetic.py        # Script to generate synthetic test data
└── README.md                    # This file
```

## File Formats

### `od_flows.csv`
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int/str | Time interval index or datetime string |
| `origin` | str | Origin tile ID |
| `destination` | str | Destination tile ID |
| `flow` | int | Number of individuals moving from origin to destination |

### `tile_features.csv`
| Column | Type | Description |
|--------|------|-------------|
| `tile_id` | str | Tile identifier |
| `population` | float | Living population of the tile |
| `accommodation` | int | Number of POIs: accommodation category |
| `active_life` | int | Number of POIs: active_life category |
| `arts_and_entertainment` | int | Number of POIs: arts_and_entertainment |
| `attractions_and_activities` | int | Number of POIs: attractions_and_activities |
| `automotive` | int | Number of POIs: automotive |
| `beauty_and_spa` | int | Number of POIs: beauty_and_spa |
| `eat_and_drink` | int | Number of POIs: eat_and_drink |
| `education` | int | Number of POIs: education |
| `financial_service` | int | Number of POIs: financial_service |
| `health_and_medical` | int | Number of POIs: health_and_medical |
| `professional_services` | int | Number of POIs: professional_services |
| `public_service_and_government` | int | Number of POIs: public_service_and_government |
| `real_estate` | int | Number of POIs: real_estate |
| `religious_organization` | int | Number of POIs: religious_organization |
| `retail` | int | Number of POIs: retail |
| `travel` | int | Number of POIs: travel |

### `tile_centroids.csv`
| Column | Type | Description |
|--------|------|-------------|
| `tile_id` | str | Tile identifier |
| `latitude` | float | Centroid latitude |
| `longitude` | float | Centroid longitude |

### `weather.csv`
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int/str | Time interval (same as od_flows.csv) |
| `precipitation_mm` | float | Precipitation in millimeters |
| `temperature_c` | float | Perceived temperature in Celsius |
| `hour_of_day` | int | Hour of the day (0-23) |

## Datasets Used in the Paper

### 1. Bike NYC (CitiBike)
- **Source**: https://ride.citibikenyc.com/system-data
- **Period**: April–September 2014
- **Grid**: 16 × 8
- **Temporal aggregation**: Hourly

### 2. Taxi Beijing (T-Drive)
- **Source**: Microsoft Research (T-Drive dataset)
- **Period**: One week, February 2008
- **Grid**: 32 × 32
- **Temporal aggregation**: 30-minute bins
- **Reference**: Yuan et al., 2011

### 3. Spain OD
- **Source**: District-level mobility data
- **Period**: February 1 – April 30, 2023
- **Tiles**: 2,735 irregular tiles
- **Temporal aggregation**: Hourly
- **Reference**: pySpainMobility

## POI Data (Overture Maps)

Points of Interest (POIs) are obtained from Overture Maps using DuckDB:
- Website: https://overturemaps.org/
- 16 top-level categories (see `tile_features.csv` format above)
- Each POI has latitude, longitude, and category
- POIs are mapped to tiles and counted per category

## Quick Start: Synthetic Data

To generate synthetic data for testing:

```bash
python data/generate_synthetic.py --output_dir data/synthetic --n_tiles 10 --n_timesteps 100
```

This creates a small synthetic dataset for pipeline verification.
