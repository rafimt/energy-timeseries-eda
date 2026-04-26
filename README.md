# energy-timeseries-eda

Exploratory data analysis and feature engineering for household appliance energy consumption forecasting. Based on the UCI Appliances Energy Prediction dataset — 10-minute interval sensor readings from a low-energy building.

---

## Project Structure

```
energy-timeseries-eda/
├── notebooks/
│   └── EDA.ipynb               # Time series analysis, seasonality, correlations
├── src/
│   └── data_processing.py      # Full processing pipeline (resampling → features → splits)
├── data/
│   ├── raw/                    # Raw CSV files (gitignored)
│   └── processed/              # Parquet outputs (gitignored)
└── outputs/
    ├── plots/                  # Saved visualizations
    └── reports/                # Summary reports
```

---

## Dataset

**UCI Appliances Energy Prediction**  
- Frequency: 10-minute intervals → resampled to hourly  
- Sensors: 9 indoor temperature + 9 humidity sensors, outdoor weather  
- Target: `Appliances` (Wh, household appliance energy consumption)

---

## Processing Pipeline

`src/data_processing.py` runs the full pipeline in sequence:

| Step | Function | Description |
|---|---|---|
| 1 | `load_raw_data` | Parse CSV with DatetimeIndex |
| 2 | `resample_to_hourly` | 10-min → 1-hour (sum energy, mean weather) |
| 3 | `consolidate_sensors` | 18 sensors → 7 aggregates (mean, std, min, max, T_diff) |
| 4 | `add_time_features` | Cyclical sin/cos encoding for hour, day-of-week, month |
| 5 | `add_lag_features` | Lag 1h/24h + rolling 6h/24h mean (leakage-safe) |
| 6 | `split_data` | Chronological 70/15/15 train-val-test split |
| 7 | `scale_data` | StandardScaler fit on train only (features + target) |

### Run the pipeline

```python
from src.data_processing import run_pipeline

summary = run_pipeline(
    raw_path="data/raw/energydata_complete.csv",
    output_dir="data/processed"
)
```

Outputs written to `data/processed/`:
- `energy_hourly_features.parquet` — full feature set
- `train.parquet`, `val.parquet`, `test.parquet` — scaled splits
- `scaler.pkl`, `target_scaler.pkl` — fitted scalers for inverse-transform

---

## EDA Highlights

- **Seasonality**: clear daily and weekly cycles in appliance usage
- **Indoor/Outdoor temperature gap** (`T_diff`) correlates with heating load
- **Lighting** (`lights`) is a strong proxy for occupancy
- **Lag features** (1h, 24h) capture autocorrelation in consumption patterns

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EDA.ipynb
```

---

## Related

- MLOps training pipeline: `../mlops-project`
