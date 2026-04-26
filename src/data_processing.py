"""
Data processing pipeline for Energy Forecast MLOps project.

Converts raw 10-minute sensor data into clean hourly features
ready for model training. Designed to be imported by notebooks
and orchestration tools (e.g., Prefect).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


def load_raw_data(path):
    """
    Load raw CSV, parse dates, and set a DatetimeIndex.

    Args:
        path (str or Path): Path to the raw CSV file. Must contain a 'date' column.

    Returns:
        pd.DataFrame: DataFrame sorted chronologically with a DatetimeIndex.
    """
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    return df


def resample_to_hourly(df):
    """
    Drop noise columns (rv1, rv2) and resample from 10-min to hourly.

    Energy columns (Appliances, lights) are summed (cumulative Wh).
    All other columns (temps, humidity, weather) are averaged (instantaneous measures).

    Args:
        df (pd.DataFrame): Raw 10-minute sensor data with a DatetimeIndex.

    Returns:
        pd.DataFrame: Hourly resampled DataFrame with rv1/rv2 removed.
    """
    df = df.drop(columns=["rv1", "rv2"], errors="ignore")

    sum_cols = ["Appliances", "lights"]
    mean_cols = [c for c in df.columns if c not in sum_cols]

    agg_rules = {col: "sum" for col in sum_cols}
    agg_rules.update({col: "mean" for col in mean_cols})

    df_hourly = df.resample("1h").agg(agg_rules)
    return df_hourly


def consolidate_sensors(df):
    """
    Reduce 9 temperature + 9 humidity sensors to meaningful aggregates.

    Creates: T_indoor_mean, T_indoor_std, T_indoor_max, T_indoor_min,
             RH_indoor_mean, RH_indoor_std, T_diff (indoor-outdoor).
    Drops the original 18 individual sensor columns.

    Args:
        df (pd.DataFrame): Hourly DataFrame containing T1–T9, RH_1–RH_9,
            and T_out columns.

    Returns:
        pd.DataFrame: DataFrame with individual sensor columns replaced by
            the 7 aggregate features listed above.
    """
    temp_cols = [f"T{i}" for i in range(1, 10)]
    hum_cols = [f"RH_{i}" for i in range(1, 10)]

    df["T_indoor_mean"] = df[temp_cols].mean(axis=1)
    df["T_indoor_std"] = df[temp_cols].std(axis=1)
    df["T_indoor_max"] = df[temp_cols].max(axis=1)
    df["T_indoor_min"] = df[temp_cols].min(axis=1)

    df["RH_indoor_mean"] = df[hum_cols].mean(axis=1)
    df["RH_indoor_std"] = df[hum_cols].std(axis=1)

    df["T_diff"] = df["T_indoor_mean"] - df["T_out"]

    df = df.drop(columns=temp_cols + hum_cols)
    return df


def add_time_features(df):
    """
    Add cyclical time encodings (sin/cos) and a weekend flag.

    Encodes hour-of-day, day-of-week, and month as sine/cosine pairs so the
    model sees the circular nature of time (e.g., hour 23 is close to hour 0).

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with 7 new columns: hour_sin, hour_cos,
            dow_sin, dow_cos, is_weekend, month_sin, month_cos.
    """
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def add_lag_features(df):
    """
    Add lag and rolling features for the target (Appliances) and T_diff.

    Uses shift(1) before rolling() to prevent target leakage —
    rolling windows use only strictly past values.
    Drops rows with NaN introduced by lagging (first ~24 rows).

    Args:
        df (pd.DataFrame): Hourly DataFrame containing at least
            'Appliances' and 'T_diff' columns.

    Returns:
        pd.DataFrame: DataFrame with 5 new lag/rolling columns and no NaN rows:
            Appliances_lag_1h, Appliances_lag_24h,
            Appliances_rolling_6h_mean, Appliances_rolling_24h_mean,
            T_diff_lag_1h.
    """
    df["Appliances_lag_1h"] = df["Appliances"].shift(1)
    df["Appliances_lag_24h"] = df["Appliances"].shift(24)
    df["Appliances_rolling_6h_mean"] = df["Appliances"].shift(1).rolling(6).mean()
    df["Appliances_rolling_24h_mean"] = df["Appliances"].shift(1).rolling(24).mean()
    df["T_diff_lag_1h"] = df["T_diff"].shift(1)

    df = df.dropna()
    return df


def split_data(df, train_ratio=0.70, val_ratio=0.15):
    """
    Perform a chronological train/val/test split without shuffling.

    The test set receives the remaining fraction (1 - train_ratio - val_ratio).
    Order is preserved so there is no data leakage across splits.

    Args:
        df (pd.DataFrame): Processed, time-ordered DataFrame.
        train_ratio (float): Proportion of rows for training. Default 0.70.
        val_ratio (float): Proportion of rows for validation. Default 0.15.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, val, test)
            DataFrames as copies of the corresponding slices.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def scale_data(train, val, test, target_col="Appliances"):
    """
    Fit a StandardScaler on training features and apply it to all splits.

    Both features AND target are now scaled with separate scalers.
    target_scaler is returned so predictions can be inverse-transformed
    back to Wh for interpretable metrics.
    Both scalers are fit only on train to prevent data leakage.

    Args:
        train (pd.DataFrame): Training split.
        val (pd.DataFrame): Validation split.
        test (pd.DataFrame): Test split.
        target_col (str): Target column to scale separately. Default 'Appliances'.

    Returns:
        tuple: (scaled_train, scaled_val, scaled_test, scaler, target_scaler)
    """
    feature_cols = [c for c in train.columns if c != target_col]

    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    # WHY: Previously target was left unscaled (raw Wh, mean~61 std~42) while
    # features were mean=0 std=1. MSE loss on this mismatch caused LSTM/Transformer
    # to collapse to predicting a constant (the mean). Scaling target fixes this.
    # target_scaler is saved separately so we can inverse-transform predictions
    # back to Wh when computing MAE/RMSE/MAPE.
    target_scaler = StandardScaler()
    target_scaler.fit(train[[target_col]])

    def _scale_split(split_df):
        scaled = split_df.copy()
        scaled[feature_cols] = scaler.transform(split_df[feature_cols])
        # v2: scale target too
        scaled[target_col] = target_scaler.transform(split_df[[target_col]])
        return scaled

    # v1 _scale_split (target was NOT scaled — caused scale mismatch in MSE loss):
    # def _scale_split(split_df):
    #     scaled = split_df.copy()
    #     scaled[feature_cols] = scaler.transform(split_df[feature_cols])
    #     return scaled

    scaled_train = _scale_split(train)
    scaled_val = _scale_split(val)
    scaled_test = _scale_split(test)

    return scaled_train, scaled_val, scaled_test, scaler, target_scaler


def run_pipeline(raw_path, output_dir):
    """
    Run the full data processing pipeline end-to-end.

    Executes all steps in order: load → resample → consolidate sensors →
    time features → lag features → split → scale. Persists all outputs to disk.

    Args:
        raw_path (str or Path): Path to the raw CSV file.
        output_dir (str or Path): Directory where processed files are written.
            Created automatically if it does not exist.

    Returns:
        dict: Summary with keys: raw_rows, hourly_rows_after_processing,
            features, train_shape, val_shape, test_shape,
            train_range, val_range, test_range.

    Side effects:
        Writes to output_dir:
            - energy_hourly_features.parquet  (full processed feature set)
            - train.parquet, val.parquet, test.parquet  (scaled splits)
            - scaler.pkl  (fitted StandardScaler)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    df = load_raw_data(raw_path)
    df = resample_to_hourly(df)
    df = consolidate_sensors(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    # Save full processed features
    df.to_parquet(output_dir / "energy_hourly_features.parquet")

    # Split and scale
    train, val, test = split_data(df)
    # v1: scaled_train, scaled_val, scaled_test, scaler = scale_data(train, val, test)
    # v2: target_scaler added (see scale_data docstring for why)
    scaled_train, scaled_val, scaled_test, scaler, target_scaler = scale_data(train, val, test)

    # Save splits
    scaled_train.to_parquet(output_dir / "train.parquet")
    scaled_val.to_parquet(output_dir / "val.parquet")
    scaled_test.to_parquet(output_dir / "test.parquet")

    # Save scalers
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # v2: save target_scaler so train.py can inverse-transform predictions to Wh
    with open(output_dir / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    summary = {
        "raw_rows": len(load_raw_data(raw_path)),
        "hourly_rows_after_processing": len(df),
        "features": len(df.columns),
        "train_shape": scaled_train.shape,
        "val_shape": scaled_val.shape,
        "test_shape": scaled_test.shape,
        "train_range": f"{scaled_train.index.min()} to {scaled_train.index.max()}",
        "val_range": f"{scaled_val.index.min()} to {scaled_val.index.max()}",
        "test_range": f"{scaled_test.index.min()} to {scaled_test.index.max()}",
    }

    print("Pipeline complete!")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary
